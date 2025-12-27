#!/bin/bash

MODE=$1

if [ "$MODE" = "scratchpad" ]; then
    echo "Building and running scratchpad..."
    clang -std=c23 -Wall -Wextra -pedantic -Itomgrad -o build/scratchpad scratchpad/scratchpad.c tomgrad/tomgrad.c -lm -g -O0
    if [ $? -eq 0 ]; then
        ./build/scratchpad
    fi
elif [ "$MODE" = "test" ]; then
    echo "Building and running tests..."
    clang -std=c23 -Wall -Wextra -pedantic -DUNITY_INCLUDE_DOUBLE -DUNITY_INCLUDE_PRINT_FORMATTED -Itests -Itomgrad -o build/tests tests/tests.c tests/unity.c tomgrad/tomgrad.c -lm
    if [ $? -eq 0 ]; then
        ./build/tests
    fi
else
    echo "Usage: $0 <scratchpad|test>"
    echo "  scratchpad - Build and run scratchpad.c"
    echo "  test       - Build and run tests.c"
    exit 1
fi
