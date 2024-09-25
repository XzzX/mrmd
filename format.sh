#! /bin/bash

git ls-files -- mrmd*.cpp mrmd*.hpp | xargs clang-format-18 -i -style=file
