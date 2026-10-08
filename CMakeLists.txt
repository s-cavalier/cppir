cmake_minimum_required(VERSION 3.16)

project(cppIR LANGUAGES CXX)

add_executable(cppIR
    src/main.cpp
)

target_compile_features(cppIR PRIVATE cxx_std_23)

