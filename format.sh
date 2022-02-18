#! /bin/bash

git ls-files -- mrmd*.cpp mrmd*.hpp | xargs clang-format -i -style=file
