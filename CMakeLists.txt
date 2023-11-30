cmake_minimum_required(VERSION 3.22)

project(nn_simple VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/sqlite3)

file(GLOB_RECURSE MY_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp )
add_executable(nn_simple ${MY_SOURCE})

target_include_directories(nn_simple  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nn_simple eigen )
target_link_libraries(nn_simple sqlite3)