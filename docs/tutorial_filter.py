#!/usr/bin/env python3
# tutorial_filter.py: turns a commented .cpp into a Doxygen page
import sys, re, os, textwrap

COMMENT = re.compile(r'^\s*//\s?(.*)$')   # whole-line // comment -> prose

path = sys.argv[1]
lines = open(path, encoding="utf-8").read().splitlines()

name = "tut_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
title = name

# Take the title only from a comment that is the very first non-blank line,
# so a comment further down (e.g. inside main) is never swallowed.
for i, line in enumerate(lines):
    if not line.strip():
        continue
    m = COMMENT.match(line)
    if m:
        title = m.group(1).strip()
        del lines[i]
    break

out = [f"/** \\page {name} {title}", ""]
code = []

def flush_code():
    # Emit the collected code as one block, without leading/trailing blank
    # lines and with common indentation removed, so every block looks the same.
    while code and not code[0].strip():
        code.pop(0)
    while code and not code[-1].strip():
        code.pop()
    if code:
        body = textwrap.dedent("\n".join(code)).replace("*/", "* /")  # don't close the outer comment
        out.extend(["", "\\code{.cpp}", body, "\\endcode", ""])
    code.clear()

for line in lines:
    m = COMMENT.match(line)
    if m:
        flush_code()
        out.append(m.group(1))
    else:
        code.append(line)
flush_code()

out.append("*/")
print("\n".join(out))
