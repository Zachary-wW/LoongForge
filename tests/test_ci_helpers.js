// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { validatePullRequestTitle } = require('../.github/scripts/pr_title');
const { filterModelsWithBaselines, missingModels } = require('../.github/scripts/ci_model_validation');
const {
  findActionRequiredCheck,
  gpuConclusion,
  hasGpuValidation,
} = require('../.github/scripts/gpu_checks');

const repoRoot = path.resolve(__dirname, '..');

function readWorkflow(name) {
  return fs.readFileSync(path.join(repoRoot, '.github/workflows', name), 'utf8');
}

test('accepts a valid multi-module title', () => {
  const result = validatePullRequestTitle('[llm, ckpt] feat: add converter');
  assert.equal(result.ok, true);
  assert.deepEqual(result.modules, ['llm', 'ckpt']);
});

test('rejects a title without a module', () => {
  const result = validatePullRequestTitle('[, ] feat: empty modules');
  assert.equal(result.ok, false);
  assert.match(result.message, /at least one valid module/i);
});

test('rejects duplicate modules', () => {
  const result = validatePullRequestTitle('[ci, ci] fix: duplicate module');
  assert.equal(result.ok, false);
  assert.match(result.message, /duplicate module/i);
});

test('rejects an unknown module', () => {
  const result = validatePullRequestTitle('[unknown] fix: unknown module');
  assert.equal(result.ok, false);
  assert.match(result.message, /invalid modules/i);
});

test('new commits invalidate suite GPU jobs without SHA-scoped concurrency', () => {
  const regression = readWorkflow('gpu-regression.yml');
  const invalidation = readWorkflow('gpu-invalidate.yml');
  assert.doesNotMatch(regression, /group: gpu-regression-.*head_sha/);
  for (const suite of ['llm_vlm', 'embodied']) {
    assert.match(regression, new RegExp(`group: gpu-regression-.*-${suite}`));
    assert.match(invalidation, new RegExp(`group: gpu-regression-.*-${suite}`));
  }
});

test('GPU invalidation recognizes a prior check or bot result comment', () => {
  assert.equal(hasGpuValidation([{ name: 'gpu-regression' }], []), true);
  assert.equal(hasGpuValidation([], [
    { user: { login: 'github-actions[bot]' }, body: '<!-- loongforge-gpu-regression:abc -->' },
  ]), true);
  assert.equal(hasGpuValidation([], [
    { user: { login: 'maintainer' }, body: '<!-- loongforge-gpu-regression:abc -->' },
  ]), false);
  assert.equal(hasGpuValidation([], []), false);
});

test('GPU invalidation reuses only an action_required check for the current SHA', () => {
  const checks = [
    { id: 1, name: 'gpu-regression', head_sha: 'old', status: 'completed', conclusion: 'action_required' },
    { id: 2, name: 'gpu-regression', head_sha: 'new', status: 'completed', conclusion: 'failure' },
    { id: 3, name: 'gpu-regression', head_sha: 'new', status: 'completed', conclusion: 'action_required' },
    { id: 4, name: 'gpu-regression', head_sha: 'new', status: 'in_progress', conclusion: null },
  ];
  assert.equal(findActionRequiredCheck(checks, 'new')?.id, 3);
  assert.equal(findActionRequiredCheck(checks, 'missing'), null);
});

test('GPU conclusion keeps cancelled runs distinct from failures', () => {
  assert.deepEqual(
    gpuConclusion({ validate: 'success', embodied: 'success', llm_vlm: 'skipped' }, 'embodied'),
    { conclusion: 'success', failed: [] },
  );
  assert.deepEqual(
    gpuConclusion({ validate: 'success', embodied: 'cancelled', llm_vlm: 'skipped' }, 'embodied'),
    { conclusion: 'cancelled', failed: ['embodied'] },
  );
  assert.deepEqual(
    gpuConclusion({ validate: 'success', embodied: 'failure', llm_vlm: 'skipped' }, 'embodied'),
    { conclusion: 'failure', failed: ['embodied'] },
  );
  assert.deepEqual(
    gpuConclusion({ validate: 'failure', embodied: 'skipped', llm_vlm: 'skipped' }, 'embodied'),
    { conclusion: 'failure', failed: ['validate', 'embodied'] },
  );
});

test('PR check exposes queued build regression and cancellation stages', () => {
  const dispatch = readWorkflow('ok-to-test.yml');
  const regression = readWorkflow('gpu-regression.yml');
  assert.match(dispatch, /title: 'GPU validation queued'/);
  assert.match(regression, /title: 'Building candidate image'/);
  assert.match(regression, /title: `Running \$\{process\.env\.SUITE\} regression`/);
  assert.match(regression, /GPU validation cancelled/);
  assert.match(regression, /if \(regressionSummary\) checkOutput\.text = regressionSummary/);
  assert.doesNotMatch(regression, /text: process\.env\.REGRESSION_SUMMARY \|\| null/);
  assert.match(regression, /finalize:\n    timeout-minutes: (?:6[1-9]|[7-9][0-9])/);
});

test('GPU status is separate from the required CPU gate', () => {
  const dispatch = readWorkflow('ok-to-test.yml');
  const regression = readWorkflow('gpu-regression.yml');
  assert.match(dispatch, /name: 'gpu-regression'/);
  assert.match(regression, /name: 'gpu-regression'/);
  assert.match(dispatch, /CI_ENABLE_LLM_VLM|ENABLE_LLM_VLM/);
  assert.doesNotMatch(dispatch, /Only the embodied suite/);
  assert.equal((regression.match(/CANDIDATE_REVISION:/g) || []).length, 2);
});

test('model validation checks the generic baseline alias', () => {
  const action = fs.readFileSync(
    path.join(repoRoot, '.github/actions/validate-models/action.yml'),
    'utf8',
  );
  assert.match(action, /baseline/);
  assert.match(action, /unknown .*model/);
});

test('model validation reports a model missing its baseline', () => {
  const requested = ['known_model', 'missing_model'];
  const validated = filterModelsWithBaselines(
    requested,
    requested,
    ['known_model'],
  );
  assert.deepEqual(missingModels(requested, validated), ['missing_model']);
});

test('llm_vlm feature gate is explicit in both trusted workflows', () => {
  assert.match(
    readWorkflow('ok-to-test.yml'),
    /request\.suite === 'llm_vlm' && process\.env\.ENABLE_LLM_VLM !== 'true'/,
  );
  assert.match(
    readWorkflow('gpu-regression.yml'),
    /process\.env\.SUITE === 'llm_vlm' && process\.env\.ENABLE_LLM_VLM !== 'true'/,
  );
});

test('manual dispatch cannot publish the required PR gate name', () => {
  const workflow = readWorkflow('static-checks.yml');
  assert.match(
    workflow,
    /github\.event_name == 'pull_request'.*'static-checks'.*'manual-static-checks'/,
  );
});

test('Claude review excludes project settings', () => {
  const workflow = readWorkflow('claude-review.yml');
  assert.match(workflow, /--setting-sources user/);
  assert.doesNotMatch(workflow, /--setting-sources project/);
});

test('Claude review uses the configured runner labels', () => {
  const workflow = readWorkflow('claude-review.yml');
  assert.match(workflow, /runs-on: \$\{\{ fromJSON\(vars\.CI_REVIEW_RUNNER\) \}\}/);
});

test('Claude review uses the configured model and reports failures', () => {
  const workflow = readWorkflow('claude-review.yml');
  assert.match(workflow, /--model "\$\{\{ vars\.CLAUDE_REVIEW_MODEL \}\}"/);
  assert.match(workflow, /path_to_claude_code_executable:/);
  assert.match(workflow, /steps\.claude\.outcome == 'failure'/);
  assert.match(workflow, /api_error_status/);
  assert.match(workflow, /terminal_reason/);
  assert.doesNotMatch(workflow, /show_full_output:\s*true/);
});

test('release uses matching context and immutable image tags', () => {
  const workflow = readWorkflow('release.yml');
  assert.match(workflow, /group: release-\$\{\{ github\.ref \}\}/);
  assert.match(workflow, /path: LoongForge/);
  assert.match(workflow, /file: LoongForge\/docker\/Dockerfile/);
  assert.match(workflow, /needs: \[validate, package, publish\]/);
  assert.match(workflow, /submodules: recursive/);
  assert.match(workflow, /LoongForge\/\*\*\/.git/);
  assert.doesNotMatch(workflow, /DOCKERHUB_IMAGE.*:latest/);
});

test('all external Actions are pinned to full commit SHAs', () => {
  for (const name of fs.readdirSync(path.join(repoRoot, '.github/workflows'))) {
    if (!name.endsWith('.yml') && !name.endsWith('.yaml')) continue;
    const workflow = readWorkflow(name);
    for (const match of workflow.matchAll(/^\s+uses:\s+([^@\s]+)@([^\s#]+)/gm)) {
      if (match[1].startsWith('./')) continue;
      assert.match(match[2], /^[0-9a-f]{40}$/, `${name}: ${match[1]} is not pinned`);
    }
  }
  const action = fs.readFileSync(
    path.join(repoRoot, '.github/actions/validate-models/action.yml'),
    'utf8',
  );
  assert.match(action, /uses:\s+actions\/github-script@[0-9a-f]{40}/);
  const sync = readWorkflow('submodule-sync.yml');
  assert.match(sync, /create-github-app-token@[0-9a-f]{40}/);
  assert.match(sync, /token: \$\{\{ steps\.app-token\.outputs\.token \}\}/);
});
