#!/usr/bin/env bash

set -e

DIR=$(dirname "$0")
echo $DIR

valgrind --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --verbose \
    --log-file="$DIR/valgrind-out.txt" \
    "$@"
