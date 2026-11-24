#!/usr/bin/env python3
"""Normalize a Python file to code-only tokens so a diff shows only REAL code
changes (Rule 2/3 live-code diff tool, project-independent — no EDIT-ME).

Strips comments, docstrings, blank lines, indentation and all line-wrapping /
reformatting differences, emitting one logical statement per line.

Why this exists: a base<->transplant file diff routinely shows 50-100 changed
lines that are entirely reflow, renamed imports, or docstrings. Normalizing both
sides first collapses those to zero and leaves only the hunks worth classifying
as dead-code / probe / live-op.

Usage:
    python codenorm.py BASE/models/foo.py  > /tmp/base_foo.norm
    python codenorm.py TRANS/model/foo.py  > /tmp/trans_foo.norm
    diff -u /tmp/base_foo.norm /tmp/trans_foo.norm

or use the diff_live_code.sh wrapper next to this file.
"""
import ast
import sys
import tokenize

path = sys.argv[1]

# Line numbers occupied by module / class / function docstrings.
src = open(path, encoding="utf-8").read()
doc_lines = set()
for node in ast.walk(ast.parse(src)):
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            n = body[0]
            doc_lines.update(range(n.lineno, n.end_lineno + 1))

SKIP = (tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
        tokenize.ENCODING, tokenize.ENDMARKER)

with open(path, "rb") as fh:
    toks = list(tokenize.tokenize(fh.readline))

out = []
for t in toks:
    if t.type in SKIP:
        continue
    if t.type == tokenize.STRING and t.start[0] in doc_lines:
        continue
    out.append("\n" if t.type == tokenize.NEWLINE else t.string)

print(" ".join(out).replace(" \n ", "\n").strip())
