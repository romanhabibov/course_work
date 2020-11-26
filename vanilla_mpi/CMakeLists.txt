cmake_minimum_required(VERSION 3.0)

project(vanilla_mpi)

find_package(MPI REQUIRED)

set(SOURCE_EXE main.c)

add_executable(main ${SOURCE_EXE})

include_directories('/usr/local/include')
link_directories('/usr/local/bin')
target_link_libraries(main mpi m)