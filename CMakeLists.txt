cmake_minimum_required(VERSION 3.9)
project(Toybrot)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/redist)


find_package(SDL2)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message("Enabling CUDA projects")
else()
    message("Couldn't find CUDA, disabling projects")
endif()

if(TARGET SDL2::SDL2)

    message("Found SDL2, enabling interactive projects")
    add_subdirectory(SDL)
    add_subdirectory(MPISDL)

    if(CMAKE_CUDA_COMPILER)
        add_subdirectory(CUDASDL)
    endif()

else()
    message("Couldn't find SDL2, disabling interactive projects")
endif()

add_subdirectory(MPI)





