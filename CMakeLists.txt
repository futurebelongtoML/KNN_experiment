cmake_minimum_required(VERSION 3.6)
project(experiment)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES KNN_test.cpp)
add_executable(KNN_test ${SOURCE_FILES})