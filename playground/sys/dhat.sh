#!/usr/bin/env bash

set -e

DIR=$(dirname "$0")
echo $DIR

valgrind --tool=dhat \
    --dhat-out-file="$DIR/dhat-out.txt" \
    "$@"
