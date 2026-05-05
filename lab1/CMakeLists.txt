cmake_minimum_required(VERSION 3.15)
project(lab1_parallel CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build with optimisations
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# OpenMP
find_package(OpenMP REQUIRED)

add_executable(lab1 main.cpp)

target_link_libraries(lab1
  PRIVATE
    OpenMP::OpenMP_CXX
    Threads::Threads
)

find_package(Threads REQUIRED)
target_link_libraries(lab1 PRIVATE Threads::Threads)
