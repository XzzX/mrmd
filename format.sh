#! /bin/bash

git ls-files -- src*.cpp src*.hpp | xargs clang-format -i -style=file
