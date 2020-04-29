 cmake_minimum_required(VERSION 3.7)
project(https://github.com/hamling-ling/OsakanaFFT.git CXX)

find_package(OpenCL REQUIRED)

add_executable(acorr acorr.cpp)
target_link_libraries(acorr OpenCL::OpenCL)

configure_file(acorr.cl acorr.cl COPYONLY)
