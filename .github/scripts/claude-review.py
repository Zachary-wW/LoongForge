#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Call Claude API to review a PR diff, then post inline review comments."""
import json
import os
import re
import subprocess
import sys
import urllib.request


def call_claude(api_key, base_url, model, prompt):
    payload = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }

    req = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())

    for block in result.get("content", []):
        if block.get("type") == "text" and "text" in block:
            return block["text"]

    print(
        f"Claude response did not include a text content block: {json.dumps(result)[:1000]}",
        file=sys.stderr,
    )
    raise RuntimeError("Claude response missing text content block")


def parse_diff_files(diff_text):
    """Extract file paths and their line ranges from a unified diff."""
    files = {}
    current_file = None
    current_line = 0

    for line in diff_text.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            files[current_file] = set()
        elif line.startswith("@@ "):
            match = re.search(r"\+(\d+)", line)
            if match:
                current_line = int(match.group(1)) - 1
        elif current_file:
            if line.startswith("+") and not line.startswith("+++"):
                current_line += 1
                files[current_file].add(current_line)
            elif line.startswith("-"):
                pass
            else:
                current_line += 1

    return files


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    model = os.environ.get("CLAUDE_MODEL", "Claude Sonnet 4.6")
    skill_path = os.environ.get("SKILL_PATH", "skills/loongforge-review/SKILL.md")
    diff_path = os.environ.get("DIFF_PATH", "/tmp/pr-diff.txt")
    pr_title = os.environ.get("PR_TITLE", "")
    pr_author = os.environ.get("PR_AUTHOR", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    with open(skill_path, "r") as f:
        skill = f.read()

    with open(diff_path, "r") as f:
        diff = f.read()

    truncated = len(diff) > 80000
    diff_trimmed = diff[:80000]

    prompt = f"""{skill}

## PR Information

- Title: {pr_title}
- Author: {pr_author}

## Diff

```diff
{diff_trimmed}
```

{"⚠️ Note: The diff was truncated to 80,000 characters. Review may be incomplete." if truncated else ""}

## Output Requirements

You MUST respond with a valid JSON object (no markdown fencing, no extra text). The JSON must have this exact structure:

{{
  "verdict": "APPROVE" | "REQUEST_CHANGES" | "COMMENT",
  "summary": "1-3 sentence overall assessment",
  "inline_comments": [
    {{
      "path": "relative/path/to/file.py",
      "line": 42,
      "body": "Issue description with severity prefix: [Critical] or [Warning] or [Suggestion]"
    }}
  ]
}}

Rules for inline_comments:
- "path" must be a file path that appears in the diff (after +++ b/)
- "line" must be a line number within a ADDED (+) hunk of the diff
- "body" should start with [Critical], [Warning], or [Suggestion] followed by the issue
- Only comment on lines that are CHANGED in this diff, not pre-existing code
- Keep each comment actionable and concise

Now review this pull request and respond with ONLY the JSON object."""

    try:
        response = call_claude(api_key, base_url, model, prompt)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"API Error ({e.code}): {body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse JSON from response (handle possible markdown fencing)
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        review = json.loads(text)
    except json.JSONDecodeError:
        print(f"Failed to parse Claude response as JSON:\n{response[:500]}", file=sys.stderr)
        # Fall back to posting raw response as comment
        with open("/tmp/review-result.json", "w") as f:
            json.dump({"verdict": "COMMENT", "summary": response[:2000], "inline_comments": []}, f)
        sys.exit(0)

    # Validate inline comments against actual diff
    diff_files = parse_diff_files(diff)
    valid_comments = []
    for comment in review.get("inline_comments", []):
        path = comment.get("path", "")
        line = comment.get("line", 0)
        if path in diff_files and line in diff_files[path]:
            valid_comments.append(comment)
        elif path in diff_files:
            # Line not in diff hunks, try to find nearest valid line
            valid_lines = sorted(diff_files[path])
            if valid_lines:
                nearest = min(valid_lines, key=lambda x: abs(x - line))
                comment["line"] = nearest
                comment["body"] = f"{comment['body']} (originally flagged at line {line})"
                valid_comments.append(comment)

    review["inline_comments"] = valid_comments

    with open("/tmp/review-result.json", "w") as f:
        json.dump(review, f, ensure_ascii=False)

    print(f"Review complete: {review['verdict']}, {len(valid_comments)} inline comments")


if __name__ == "__main__":
    main()
