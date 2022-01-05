#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from jinja2 import Environment, FileSystemLoader
import json
from pathlib import Path
from pprint import pprint
import os

def cap_first(s):
    return s[0].capitalize() + s[1:]


def get_jinja_environment():
    dirname = os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(dirname))
    env.filters['cap_first'] = cap_first
    return env


def generate_file(path, template, context={}, filename=None):
    path = Path(path)
    if filename is None:
        filename = Path(template.replace(".jinja2", ""))
    env = get_jinja_environment()
    print(f"generating: {(path / filename)}")
    with open(path / filename, "wb") as fout:
        content = env.get_template(template).render(context)
        fout.write(content.encode('utf8'))

with open('DumpH5MDParallel.json.jinja2') as f:
    context = json.load(f)

pprint(context)

generate_file('.', 'DumpH5MDParallel.cpp.jinja2', context)
generate_file('.', 'DumpH5MDParallel.hpp.jinja2', context)

generate_file('.', 'RestoreH5MDParallel.cpp.jinja2', context)
generate_file('.', 'RestoreH5MDParallel.hpp.jinja2', context)

