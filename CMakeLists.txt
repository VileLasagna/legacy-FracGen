cmake_minimum_required(VERSION 2.8)
project(Toybrot)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/redist)

add_executable(${PROJECT_NAME}          "Toybrot/main.cpp"
                                        "Toybrot/FracGenWindow.h"
                                        "Toybrot/FracGenWindow.cpp")

add_executable(${PROJECT_NAME}MPI       "MPI/main.cpp"
                                        "MPI/pngWriter.h"
                                        "MPI/pngWriter.cpp")

add_executable(${PROJECT_NAME}MPISDL    "MPISDL/main.cpp" )


target_compile_features(${PROJECT_NAME}MPI PUBLIC cxx_std_14)


target_link_libraries(${PROJECT_NAME}       SDL2)
target_link_libraries(${PROJECT_NAME}MPISDL SDL2)


if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME}    pthread)
    target_link_libraries(${PROJECT_NAME}MPI pthread)
endif()

install(TARGETS ${PROJECT_NAME}       RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}MPI    RUNTIME DESTINATION bin)
install(TARGETS ${PROJECT_NAME}MPISDL RUNTIME DESTINATION bin)


find_package(MPI)
find_package(PNG)


if(TARGET MPI::MPI_CXX)
    target_link_libraries(${PROJECT_NAME}MPI    MPI::MPI_CXX)
    target_link_libraries(${PROJECT_NAME}MPISDL MPI::MPI_CXX)
endif()

if(TARGET PNG::PNG)
    target_link_libraries(${PROJECT_NAME}MPI PNG::PNG)
endif()
