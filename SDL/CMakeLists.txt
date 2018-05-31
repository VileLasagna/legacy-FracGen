project(ToybrotSDL LANGUAGES CXX)

add_executable(${PROJECT_NAME}  "main.cpp"
                                "FracGenWindow.hpp"
                                "FracGenWindow.cpp")

if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

target_link_libraries(${PROJECT_NAME} SDL2::SDL2)

