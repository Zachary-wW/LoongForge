#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Call Claude API to review a PR diff and output the review as markdown."""
import json
import os
import sys
import urllib.request


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    model = os.environ.get("CLAUDE_MODEL", "Claude Sonnet 4.6")
    skill_path = os.environ.get("SKILL_PATH", "skills/loongforge-review/SKILL.md")
    diff_path = os.environ.get("DIFF_PATH", "/tmp/pr-diff.txt")
    pr_title = os.environ.get("PR_TITLE", "")
    pr_author = os.environ.get("PR_AUTHOR", "")

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    with open(skill_path, "r") as f:
        skill = f.read()

    with open(diff_path, "r") as f:
        diff = f.read()[:80000]

    prompt = f"""{skill}

## PR Information

- Title: {pr_title}
- Author: {pr_author}

## Diff

```diff
{diff}
```

Now review this pull request following the instructions above."""

    payload = {
        "model": model,
        "max_tokens": 4096,
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

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
            text = result["content"][0]["text"]
            print(text)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"API Error ({e.code}): {body}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
