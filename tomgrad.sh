#!/bin/bash
set -e

BUILD_DIR="build"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Debug}"
COMMAND="${1:-build}"

case "$COMMAND" in
    build|compile)
        echo "Building tomgrad (${CMAKE_BUILD_TYPE})..."

        if [ ! -d "$BUILD_DIR" ]; then
            mkdir "$BUILD_DIR"
        fi

        cd "$BUILD_DIR"
        cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" ..

        if [ -f "compile_commands.json" ]; then
            ln -sf "$BUILD_DIR/compile_commands.json" "../compile_commands.json"
        fi

        cmake --build . --parallel $(sysctl -n hw.ncpu 2>/dev/null || echo 4)

        if [ "$COMMAND" = "compile" ] && [ -f "../examples/example.cpp" ]; then
            clang++ -std=c++23 -Wall -Wextra -Werror \
                -I../include \
                ../examples/example.cpp \
                -o example
            echo "Compilation complete! Run ./build/example to execute"
        else
            echo "Build complete!"
        fi
        ;;

    test)
        echo "Running tests for tomgrad (${CMAKE_BUILD_TYPE})..."

        if [ ! -d "$BUILD_DIR" ]; then
            echo "Build directory not found. Running build first..."
            "$0" build
        fi

        cd "$BUILD_DIR"
        ctest --output-on-failure --verbose

        echo "Tests complete!"
        ;;

    scratchpad)
        if [ ! -d "$BUILD_DIR" ]; then
            mkdir "$BUILD_DIR"
        fi

        cd "$BUILD_DIR"

        clang++ -std=c++23 -Wall -Wextra \
            -I../include \
            ../scratchpad/scratchpad.cpp \
            -o scratchpad


        ./scratchpad
        ;;

    clean)
        echo "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        echo "Clean complete!"
        ;;

    *)
        echo "Usage: $0 {build|compile|test|scratchpad|clean}"
        echo ""
        echo "Commands:"
        echo "  build      - Build the project (default)"
        echo "  compile    - Build and compile examples"
        echo "  test       - Run tests"
        echo "  scratchpad - Compile scratchpad program"
        echo "  clean      - Remove build directory"
        echo ""
        echo "Environment variables:"
        echo "  CMAKE_BUILD_TYPE - Debug (default) or Release"
        exit 1
        ;;
esac
