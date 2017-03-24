cmake_minimum_required(VERSION 2.8)
project(FracGenCMake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

#set(CMAKE_CXX_WARNINGS -Wall -Wno-ignored-qualifiers -Wno-missing-braces -Wno-unknown-pragmas -Wno-macro-redefined -Wno-microsoft-exception-spec -Wno-missing-variable-declarations  )

#set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_WARNINGS})

#include (FindSDL)


add_executable(${PROJECT_NAME} "FracGen/main.cpp")
#target_link_libraries(FracGenCMake ${SDL_LIBRARY} pthread)
target_link_libraries(FracGenCMake SDL2 pthread)
