cmake_minimum_required(VERSION 2.8)
project(FracGenCMake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/redist)

add_executable(${PROJECT_NAME} "FracGen/main.cpp")

target_link_libraries(FracGenCMake SDL2)

if(NOT WIN32)
    target_link_libraries(FracGenCMake pthread)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
