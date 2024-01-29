#!/usr/bin/env bash

set -e

DIR=$(dirname "$0")
echo $DIR

valgrind --tool=callgrind \
    --callgrind-out-file="$DIR/callgrind-out.txt" \
    "$@"
