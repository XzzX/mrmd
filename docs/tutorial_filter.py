#!/usr/bin/env python3
# tutorial_filter.py: turns a commented .cpp into a Doxygen page
#
# Lines starting with the prose marker (default "//>") become text between the
# code blocks. All other lines, including normal // and /* */ comments and doc
# comments, stay in the code exactly as written.
#
# The first non-blank line becomes the page title if it is a prose line.
#
# The page is emitted as /// line comments rather than one /** ... */ block,
# so /** ... */ doc comments inside the tutorial code can't end the page early.
import sys, re, os, textwrap

MARKER = "//>"   # change this to use a different prose marker

PROSE = re.compile(r"^\s*" + re.escape(MARKER) + r">* ?(.*)$")  # ">*": tolerate a doubled marker like "//>>"

path = sys.argv[1]
lines = open(path, encoding="utf-8").read().splitlines()

name = "tut_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
title = name

# Take the title only from a prose line that is the very first non-blank line,
# so prose further down is never swallowed.
for i, line in enumerate(lines):
    if not line.strip():
        continue
    m = PROSE.match(line)
    if m:
        title = m.group(1).strip()
        del lines[i]
    break

out = [f"\\page {name} {title}", ""]
code = []

def flush_code():
    # Emit the collected code as one block, without leading/trailing blank
    # lines and with common indentation removed, so every block looks the same.
    while code and not code[0].strip():
        code.pop(0)
    while code and not code[-1].strip():
        code.pop()
    if code:
        body = textwrap.dedent("\n".join(code)).split("\n")
        out.extend(["", "\\code{.cpp}", *body, "\\endcode", ""])
    code.clear()

for line in lines:
    m = PROSE.match(line)
    if m:
        flush_code()
        out.append(m.group(1).rstrip())
    else:
        code.append(line)
flush_code()

# Prefix every line (also blank ones) so it stays one contiguous comment block
print("\n".join("/// " + l if l else "///" for l in out))
