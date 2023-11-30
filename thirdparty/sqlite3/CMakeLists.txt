cmake_minimum_required(VERSION 3.22)
project(sqlite3)
add_library(sqlite3)
target_include_directories(sqlite3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SQLITE3_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

target_sources(sqlite3 PRIVATE ${SQLITE3_SOURCES})