cmake_minimum_required(VERSION 3.20)
project(tomgrad VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(Catch2)

include_directories(${PROJECT_SOURCE_DIR}/include)

option(BUILD_TESTING "Build tests" ON)

if(BUILD_TESTING)
    enable_testing()
    
    add_executable(tests
        tests/test_main.cpp
    )
    
    target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
    target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
    
    add_test(NAME tests COMMAND tests)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Building tests: ${BUILD_TESTING}")
