#!/bin/bash

MODE=$1

CC=clang
STD=-std=c23
WARN_FLAGS="-Wall -Wextra -pedantic"
INCLUDE_FLAGS="-Itomgrad"

DEBUG_FLAGS="-gdwarf-4 -O0 -fno-omit-frame-pointer"
ASAN_FLAGS="-fsanitize=address -fno-common"
SLOW_DEBUG_FLAGS="$DEBUG_FLAGS $ASAN_FLAGS"

LIBS="-lm"

UNITY_FLAGS="-DUNITY_INCLUDE_DOUBLE -DUNITY_INCLUDE_PRINT_FORMATTED -Itests"

if [ "$MODE" = "scratchpad" ]; then
    echo "Building and running scratchpad..."
    $CC $STD $WARN_FLAGS $INCLUDE_FLAGS -o build/scratchpad scratchpad.c $LIBS $SLOW_DEBUG_FLAGS
    if [ $? -eq 0 ]; then
        ./build/scratchpad
    fi
elif [ "$MODE" = "test" ]; then
    echo "Building and running tests..."
    $CC $STD $WARN_FLAGS $UNITY_FLAGS $INCLUDE_FLAGS -o build/tests tests.c unity/unity.c $LIBS $SLOW_DEBUG_FLAGS
    if [ $? -eq 0 ]; then
        ./build/tests
    fi
elif [ "$MODE" = "valgrind-scratchpad" ]; then
    echo "Building and running scratchpad with valgrind..."
    $CC $STD $WARN_FLAGS $INCLUDE_FLAGS -o build/scratchpad scratchpad/scratchpad.c $LIBS $SLOW_DEBUG_FLAGS
    if [ $? -eq 0 ]; then
        valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./build/scratchpad
    fi
elif [ "$MODE" = "valgrind-test" ]; then
    echo "Building and running tests with valgrind..."
    $CC $STD $WARN_FLAGS $UNITY_FLAGS $INCLUDE_FLAGS -o build/tests tests.c unity/unity.c $LIBS $SLOW_DEBUG_FLAGS
    if [ $? -eq 0 ]; then
        valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./build/tests
    fi
else
    echo "Usage: $0 <scratchpad|test|valgrind-scratchpad|valgrind-test>"
    echo "  scratchpad           - Build and run scratchpad.c"
    echo "  test                 - Build and run tests.c"
    echo "  valgrind-scratchpad  - Build and run scratchpad.c with valgrind"
    echo "  valgrind-test        - Build and run tests.c with valgrind"
    exit 1
fi
