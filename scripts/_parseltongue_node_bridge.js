#!/usr/bin/env node
/**
 * Node bridge for P4RS3LT0NGV3 transforms.
 *
 * Reads a JSON payload from stdin with shape:
 *   { transforms: ["leetspeak", "zalgo", ...], examples: [{id, text}, ...] }
 *
 * Writes to stdout:
 *   { results: [{id, text, transform, output}, ...], errors: [{transform, message}, ...] }
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
        const output = tx.func(String(ex.text || ''));
        results.push({
          id: ex.id,
          text: ex.text,
          transform: key,
          output,
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
