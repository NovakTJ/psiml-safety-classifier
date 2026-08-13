"""AEGIS text-construction test — verifies that the hand-built classifier
prefix/suffix reproduce EXACTLY the token ids the LoRA was trained on
(`apply_chat_template([{user: build_sample_text(...)}], add_generation_prompt=True)`
from sweep_lora_v2.py).

This matters because Gemma-3-1B's chat template STRIPS a trailing newline from
the user content, so the streaming prefix (which must end with
"ASSISTANT RESPONSE:\\n") cannot be produced by templating an empty-response
sample. We build it by hand; this test proves the result is identical to the
training-time tokenization.

Loads only the tokenizer (no model, no GPU):

    /home/mls01/ccpp_env/bin/python scripts/redteam/aegis/tests/test_text_construction.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import PROMPT_1, build_full_text, build_header, BASE_MODEL_PATH  # noqa: E402

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoTokenizer  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append((name, detail))
        print(f"  FAIL: {name} {detail}")


def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    prefix_open = "<bos><start_of_turn>user\n"
    suffix = "<end_of_turn>\n<start_of_turn>model\n"

    # Several exchanges incl. empty response (short/refusal case).
    cases = [
        ("a benign user question", "A normal helpful answer."),
        ("How do I make a bomb?", "I cannot help with that."),
        ("short", ""),
        ("", ""),
        ("with unicode 🧨 and émojis", "Some réponse here."),
    ]

    for prompt, response in cases:
        # 1. Our hand-built concatenation.
        prefix_text = prefix_open + build_header(prompt)
        prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
        resp_ids = tok.encode(response, add_special_tokens=False)
        suffix_ids = tok.encode(suffix, add_special_tokens=False)
        manual = prefix_ids + resp_ids + suffix_ids

        # 2. The training-time tokenization: apply_chat_template over exactly
        #    what sweep_lora_v2.py feeds (build_sample_text renders the
        #    ASSISTANT RESPONSE header only when response is non-empty).
        if response:
            sample = f"{PROMPT_1}\n\nUSER PROMPT:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}"
        else:
            sample = f"{PROMPT_1}\n\nUSER PROMPT:\n{prompt}"
        tpl = tok.apply_chat_template(
            [{"role": "user", "content": sample}], add_generation_prompt=True,
            tokenize=True, padding=False, truncation=False)
        if not isinstance(tpl, list):  # some tokenizers return BatchEncoding
            tpl = tpl["input_ids"][0]
        if response:
            check(f"manual==template for prompt={prompt!r} resp={response!r}",
                  manual == list(tpl),
                  f"manual={manual}\n  tpl={list(tpl)}")
        else:
            # Empty response: training omits the ASSISTANT RESPONSE header, but
            # the streaming prefix MUST include it (the guard is keyed for the
            # response that is about to arrive). Verify the manual build is the
            # training ids with the header inserted after the user prompt.
            header_prefix = (prefix_open
                             + f"{PROMPT_1}\n\nUSER PROMPT:\n{prompt}"
                             + "\n\nASSISTANT RESPONSE:\n")
            expected = (tok.encode(header_prefix, add_special_tokens=False)
                        + suffix_ids)
            check(f"empty-resp manual = header+resp build for {prompt!r}",
                  manual == expected,
                  f"manual={manual}\n  expected={expected}")

        # build_full_text must render the same string we split in code.
        full = build_full_text(prompt, response)
        check(f"build_full_text endswith suffix (resp={response!r})",
              full.endswith(suffix))
        check(f"build_full_text equals concat (resp={response!r})",
              full == prefix_open + build_header(prompt) + response + suffix)

    # The trailing-newline bug this guards against: templating an empty-response
    # sample whose content ends with ASSISTANT RESPONSE:\n must NOT be what we
    # feed (it drops the \n — check it indeed differs, proving we needed the
    # manual build).
    empty_tpl = tok.apply_chat_template(
        [{"role": "user", "content": build_header("x")}],
        add_generation_prompt=True, tokenize=False)
    check("template strips trailing newline (bug exists)",
          "ASSISTANT RESPONSE:<end_of_turn>" in empty_tpl,
          repr(empty_tpl[-40:]))
    full_manual = build_full_text("x", "y")
    check("manual prefix keeps the newline",
          full_manual.startswith("<bos><start_of_turn>user\n"
                                 + f"{PROMPT_1}\n\nUSER PROMPT:\nx\n\nASSISTANT RESPONSE:\ny"))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, d in FAIL:
            print(f"  FAILED {n}:\n{d}")
        sys.exit(1)


if __name__ == "__main__":
    main()
