cmake_minimum_required(VERSION 2.6)
project(ACI)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

add_subdirectory(aci)