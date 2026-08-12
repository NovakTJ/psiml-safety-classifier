#!/usr/bin/env node
/**
 * Node bridge for P4RS3LT0NGV3 transforms.
 *
 * Reads a JSON payload from stdin with shape:
 *   { transforms: ["leetspeak", "zalgo", ...], examples: [{id, text}, ...] }
 *
 * Each example may either be the legacy single-text shape {id, text} or the
 * multi-field shape {id, fields: {prompt, response, ...}}; in the latter case
 * every field is transformed with the SAME transform.
 *
 * Writes to stdout:
 *   { results: [{id, text, transform, outputs: {field: output}}, ...],
 *     errors:  [{transform, message}, ...] }
 */

const path = require('path');

function findRepo() {
  // Look relative to this script first, then cwd.
  const candidates = [
    path.join(__dirname, '..', 'vendor', 'P4RS3LT0NGV3'),
    path.join(process.cwd(), 'vendor', 'P4RS3LT0NGV3'),
    path.join(process.cwd(), 'P4RS3LT0NGV3'),
  ];
  for (const dir of candidates) {
    try {
      require.resolve(path.join(dir, 'src', 'transformers', 'loader-node.js'));
      return dir;
    } catch (e) {
      // continue
    }
  }
  throw new Error(
    'Could not find P4RS3LT0NGV3 repo. Clone it to vendor/P4RS3LT0NGV3 or run from the project root.'
  );
}

const repoRoot = findRepo();
const transforms = require(path.join(repoRoot, 'src', 'transformers', 'loader-node.js'));

function run(payload) {
  const requested = payload.transforms || [];
  const examples = payload.examples || [];
  const results = [];
  const errors = [];

  for (const key of requested) {
    const tx = transforms[key];
    if (!tx || typeof tx.func !== 'function') {
      errors.push({ transform: key, message: `Transform "${key}" not found or has no func()` });
      continue;
    }
    for (const ex of examples) {
      try {
        // New shape: { id, fields: { prompt, response, ... } }; apply the same
        // transform to every field. Legacy shape { id, text } still works.
        const fields = ex.fields || (ex.text !== undefined ? { text: ex.text } : {});
        const outputs = {};
        for (const [name, value] of Object.entries(fields)) {
          outputs[name] = tx.func(String(value == null ? '' : value));
        }
        results.push({
          id: ex.id,
          text: ex.text,
          transform: key,
          outputs,
        });
      } catch (err) {
        errors.push({ transform: key, id: ex.id, message: err.message });
      }
    }
  }

  return { results, errors };
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  input += chunk;
});
process.stdin.on('end', () => {
  try {
    const payload = input ? JSON.parse(input) : {};
    const result = run(payload);
    process.stdout.write(JSON.stringify(result, null, 2));
  } catch (err) {
    process.stderr.write(err.message + '\n');
    process.exit(1);
  }
});
