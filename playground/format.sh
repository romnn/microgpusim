#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

DIR=$(realpath $(dirname "$0"))
cd "$DIR"

# clang-format \
# 	-i -style=WebKit \
# 	$DIR/src/**/*.cc \
# 	$DIR/src/**/*.hpp \
# 	$DIR/src/**/*.cpp \
# 	$DIR/src/**/*.h
