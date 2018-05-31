project(ToyBrotMPI LANGUAGES CXX)

add_executable(${PROJECT_NAME}  "main.cpp"
                                "pngWriter.hpp"
                                "pngWriter.cpp")


find_package(MPI)
find_package(PNG)


if(TARGET MPI::MPI_CXX)
    target_link_libraries(${PROJECT_NAME} MPI::MPI_CXX)
endif()

if(TARGET PNG::PNG)
    target_link_libraries(${PROJECT_NAME} PNG::PNG)
endif()

if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
