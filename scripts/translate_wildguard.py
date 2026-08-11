#!/usr/bin/env python
"""Translate WildGuard examples via a cheap LLM using OpenRouter.

Design notes
------------
- Each example is translated ONCE, into a single language selected at random
  (seeded) from a pool of languages (default: the 30-language list).
- Translates the *prompt* and *response* of each example in a single API call.
- Writes one JSONL row per example.
- Keeps the original record untouched and adds translation + metadata columns:
    original_idx, source_split, language, encoding_type, translation_model,
    prompt_template_version, timestamp, verified_accurate_description,
    augmentation_pipeline_version, notes
- `verified_accurate_description` is hardcoded to False (translations need review).
- Supports resuming: already-present (original_idx, language) rows in the output
  file are skipped on re-run.
- Failures (refusals, JSON parse failures, API errors) go to a separate
  failures file and are NOT written to the main output.

Usage
-----
    # smoke test: 4 examples into a random language drawn from a 2-language pool
    .venv/bin/python scripts/translate_wildguard.py --n-examples 4 --languages es,hi

    # full run: 1000 examples, one random language each (out of all 30)
    .venv/bin/python scripts/translate_wildguard.py --n-examples 1000

    # use the cheaper v4 flash model instead
    .venv/bin/python scripts/translate_wildguard.py --model deepseek/deepseek-v4-flash-latest
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Constants -------------------------------------------------------------

DEFAULT_LANGUAGES = [
    "zh", "es", "hi", "ar", "fr", "ru", "pt", "de", "ja", "ko",
    "it", "tr", "vi", "pl", "nl", "id", "bn", "ur", "te", "ta",
    "th", "sw", "uk", "cs", "el", "he", "fa", "mr", "ml", "hr",
]

LANGUAGE_NAMES = {
    "zh": "Chinese (Simplified)", "es": "Spanish", "hi": "Hindi", "ar": "Arabic",
    "fr": "French", "ru": "Russian", "pt": "Portuguese", "de": "German",
    "ja": "Japanese", "ko": "Korean", "it": "Italian", "tr": "Turkish",
    "vi": "Vietnamese", "pl": "Polish", "nl": "Dutch", "id": "Indonesian",
    "bn": "Bengali", "ur": "Urdu", "te": "Telugu", "ta": "Tamil",
    "th": "Thai", "sw": "Swahili", "uk": "Ukrainian", "cs": "Czech",
    "el": "Greek", "he": "Hebrew", "fa": "Persian", "mr": "Marathi",
    "ml": "Malayalam", "hr": "Croatian",
}

DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
TEMPLATE_VERSION = "v1"
PIPELINE_VERSION = "v1"
DATASET = "allenai/wildguardmix"
CONFIG = "wildguardtrain"
SPLIT = "train"


# --- Prompt template -------------------------------------------------------

def build_messages(language_name: str, prompt_text: str, response_text: str) -> list[dict[str, str]]:
    system = (
        f"You are a faithful, professional translator. Translate text from English into "
        f"{language_name}.\n"
        "Rules:\n"
        "- Translate everything. Preserve meaning, tone, formatting, code blocks, and numbers exactly.\n"
        "- Do not add, remove, soften, or censor any content. Do not refuse to translate.\n"
        "- If a field is empty, return an empty string for it.\n"
        '- Respond ONLY with a raw JSON object in this exact shape: {"prompt": "...", "response": "..."}\n'
        "- No preamble, no code fences, no extra keys."
    )
    user = f"<prompt>\n{prompt_text}\n</prompt>\n\n<response>\n{response_text}\n</response>"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_json_output(text: str) -> dict[str, Any] | None:
    """Robustly parse the model's JSON reply (handles fences and stray text)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Fallback: grab the first { ... } block.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _extract(obj: dict[str, Any], *keys: str) -> str:
    for key in keys:
        val = obj.get(key)
        if val is not None:
            return str(val)
    return ""


# --- OpenRouter client -----------------------------------------------------

def call_model(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_retries: int = 4,
    timeout: int = 180,
) -> str:
    """Call the OpenRouter chat completions endpoint with retries/backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    attempt = 0
    while True:
        try:
            resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"network error after {attempt} attempts: {exc}") from exc
            time.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("API returned no choices")
            return (choices[0].get("message") or {}).get("content") or ""

        # Retryable errors
        if resp.status_code in (429, 500, 502, 503, 504) or resp.status_code >= 500:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"HTTP {resp.status_code} after {attempt} attempts: {resp.text[:500]}")
            try:
                retry_after = float(resp.headers.get("Retry-After", "0"))
            except ValueError:
                retry_after = 0
            time.sleep(retry_after or min(2 ** attempt, 30))
            continue

        # Non-retryable
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")


def is_noop_translation(src: str, out: str) -> bool:
    """True if the model returned the source text unchanged / near-unchanged
    (a common silent-refusal pattern). Only reliable for Latin-script text;
    script changes (e.g. EN->HI) have ~0 token overlap."""
    a, b = src.strip(), out.strip()
    if not a or not b:
        return False
    if a == b:
        return True
    ta = set(re.findall(r"[A-Za-z0-9]+", a.lower()))
    tb = set(re.findall(r"[A-Za-z0-9]+", b.lower()))
    if not ta or not tb or len(ta) <= 3:
        return False
    jac = len(ta & tb) / len(ta | tb)
    return jac > 0.85


def translate_example(
    api_key: str,
    model: str,
    idx: int,
    rec: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    """Translate one (example, language) pair; returns the output row or raises.

    Empty/unparseable outputs and no-op (silent refusal) results are retried a
    few times before giving up — empty replies are usually rate-limit artifacts.
    """
    language_name = LANGUAGE_NAMES[language]
    prompt_text = rec["prompt"]
    response_text = rec["response"] or ""

    last_err: RuntimeError | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(2 * attempt)
        try:
            messages = build_messages(language_name, prompt_text, response_text)
            raw = call_model(api_key, model, messages)
            obj = parse_json_output(raw)
            if obj is None:
                raise RuntimeError(
                    f"could not parse JSON from model output (len={len(raw)}): {raw[:200]!r}"
                )
            translated_prompt = _extract(obj, "prompt", "Prompt", "PROMPT", "translated_prompt")
            translated_response = _extract(obj, "response", "Response", "RESPONSE", "translated_response")
            if not translated_prompt:
                raise RuntimeError(f"model returned empty 'prompt': {raw[:200]!r}")
            # Never trust the model for fields that were empty in the source.
            if not response_text:
                translated_response = ""

            # Silent-refusal detection: model echoes the source instead of translating.
            if is_noop_translation(prompt_text, translated_prompt):
                raise RuntimeError(
                    "no-op translation (model returned source text unchanged); raw=" + raw[:200]
                )

            out = dict(rec)
            out["translated_prompt"] = translated_prompt
            out["translated_response"] = translated_response
            out["original_idx"] = idx
            out["source_split"] = SPLIT
            out["language"] = language
            out["encoding_type"] = "none"
            out["translation_model"] = model
            out["prompt_template_version"] = TEMPLATE_VERSION
            out["timestamp"] = datetime.now(timezone.utc).isoformat()
            out["verified_accurate_description"] = False
            out["augmentation_pipeline_version"] = PIPELINE_VERSION
            notes = []
            if not rec.get("response"):
                notes.append("no_response")
            if not translated_response:
                notes.append("empty_translated_response")
            elif response_text and is_noop_translation(response_text, translated_response):
                notes.append("noop_response_translation")
            out["notes"] = ";".join(notes)
            return out
        except RuntimeError as exc:
            last_err = exc
            # No point retrying when the HTTP layer already exhausted retries
            # (that error is a RuntimeError too, but retrying here is harmless
            # and often succeeds once load drops).
            continue
    assert last_err is not None
    raise last_err


# --- Data loading / sampling -----------------------------------------------

def load_and_sample(
    n_examples: int,
    harmful_frac: float,
    seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Load the train split and sample indices (stratified by prompt_harm_label)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, CONFIG, split=SPLIT)
    harm_mask = [x == "harmful" for x in ds["prompt_harm_label"]]
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    n_harm = round(n_examples * harmful_frac)
    n_ben = n_examples - n_harm
    selected = [i for i in idxs if harm_mask[i]][:n_harm] + [i for i in idxs if not harm_mask[i]][:n_ben]
    # keep deterministic order: sort by original index
    selected.sort()
    records = [dict(ds[i]) for i in selected]
    return selected, records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_done_keys(path: Path) -> set[tuple[int, str]]:
    """Rows already present in the output file: {(original_idx, language), ...}."""
    done: set[tuple[int, str]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "original_idx" in rec and "language" in rec:
                done.add((rec["original_idx"], rec["language"]))
    return done


# --- CLI -------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-examples", type=int, default=1000, help="Examples to sample (default: 1000).")
    parser.add_argument("--harmful-frac", type=float, default=0.5, help="Fraction of sample that is harmful (default: 0.5).")
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES), help="Comma-separated ISO codes to draw from; each example gets ONE random language (default: all 30).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model id (default: {DEFAULT_MODEL}).")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "translated.jsonl", help="Output JSONL (default: data/translated.jsonl).")
    parser.add_argument("--failures", type=Path, default=PROJECT_ROOT / "data" / "translation_failures.jsonl", help="Failures JSONL (default: data/translation_failures.jsonl).")
    parser.add_argument("--sample-jsonl", type=Path, default=None, help="If set, write the sampled original rows here (useful for the parseltongue step).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent API calls (default: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42).")
    return parser.parse_args(argv)


def load_api_key(env_path: Path | None = None) -> str:
    """Load OPENROUTER_API_KEY from env, falling back to a local .env file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = env_path or PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("error: OPENROUTER_API_KEY not set (export it or create .env)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = load_api_key()
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    unknown = [lang for lang in languages if lang not in LANGUAGE_NAMES]
    if unknown:
        print(f"error: unknown language codes: {unknown}", file=sys.stderr)
        return 1

    print(f"loading dataset {DATASET}/{CONFIG} ...", file=sys.stderr)
    t0 = time.time()
    selected, records = load_and_sample(args.n_examples, args.harmful_frac, args.seed)
    print(f"sampled {len(records)} examples in {time.time()-t0:.1f}s (harmful={sum(1 for r in records if r['prompt_harm_label']=='harmful')}, unharmful={sum(1 for r in records if r['prompt_harm_label']!='harmful')})", file=sys.stderr)

    if args.sample_jsonl:
        args.sample_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.sample_jsonl, records)
        print(f"wrote sampled originals -> {args.sample_jsonl}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)

    # Resume support. Each example is translated ONCE, into one language
    # assigned deterministically from the pool (seed-stable), not every language.
    done = read_done_keys(args.output)
    # Prune stale failure records (rows that succeeded on a previous run).
    if args.failures.exists():
        stale = 0
        kept = []
        for line in args.failures.read_text(encoding="utf-8").splitlines():
            try:
                f = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (f.get("original_idx"), f.get("language")) in done:
                stale += 1
            else:
                kept.append(f)
        if stale:
            with args.failures.open("w", encoding="utf-8") as fh:
                for f in kept:
                    fh.write(json.dumps(f, ensure_ascii=False) + "\n")
            print(f"pruned {stale} stale failure records", file=sys.stderr)
    lang_rng = random.Random(f"seed{args.seed}:assign-language")
    assigned = {idx: lang_rng.choice(languages) for idx in selected}
    tasks = [(idx, rec, assigned[idx]) for idx, rec in zip(selected, records)
             if (idx, assigned[idx]) not in done]
    print(f"tasks: {len(selected)} examples, 1 language each -> {len(tasks)} calls "
          f"({len(done)} already done)", file=sys.stderr)
    if not tasks:
        print("nothing to do", file=sys.stderr)
        return 0

    # Rough cost estimate (tokens ~ chars/4; input includes template overhead)
    total_chars = sum(len(r["prompt"]) + len(r["response"] or "") for r in records)
    est_in_tokens = int(total_chars / 4) + 80 * len(records)
    est_out_tokens = int(total_chars / 4)
    est_total_tokens = est_in_tokens + est_out_tokens
    print(
        f"rough estimate: ~{est_total_tokens/1e6:.1f}M tokens across {len(tasks)} calls "
        f"(model={args.model})", file=sys.stderr
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()

    def worker(item: tuple[int, dict[str, Any], str]) -> dict[str, Any]:
        idx, rec, lang = item
        try:
            return translate_example(api_key, args.model, idx, rec, lang)
        except Exception as exc:  # noqa: BLE001 - record any failure
            return {"ok": False, "item": item, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, t) for t in tasks]
        with tqdm(total=len(tasks), desc="translating", unit="call", file=sys.stderr) as pbar:
            for fut in as_completed(futures):
                out = fut.result()
                if out.get("ok", True):
                    results.append(out)
                else:
                    idx, rec, lang = out["item"]
                    failures.append({
                        "original_idx": idx,
                        "language": lang,
                        "prompt": rec["prompt"],
                        "response": rec.get("response"),
                        "prompt_harm_label": rec["prompt_harm_label"],
                        "error": out["error"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                pbar.update(1)

    # Append results to output (resume-friendly)
    with args.output.open("a", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with args.failures.open("a", encoding="utf-8") as f:
        for rec in failures:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = time.time() - started
    print(f"done in {elapsed:.0f}s: {len(results)} ok, {len(failures)} failed", file=sys.stderr)
    if failures:
        print(f"failures logged -> {args.failures}", file=sys.stderr)
        for f_ in failures[:5]:
            print(f"  idx={f_['original_idx']} lang={f_['language']}: {f_['error'][:120]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())