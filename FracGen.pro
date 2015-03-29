
CONFIG += app
CONFIG -= qt

CONFIG(release, debug|release) \
{
    DESTDIR = $$_PRO_FILE_PWD_/distro/release
}


CONFIG(debug, debug|release) \
{
    DESTDIR = $$_PRO_FILE_PWD_/distro/debug
}

SOURCES += \
    FracGen/main.cpp

linux-g++:contains(QMAKE_HOST.arch, x86_64):\
{
    # include SDL
    LIBS += -L/usr/lib64 -lSDL2 -lSDL2_image -lSDL2_ttf -lSDL2_gfx -lSDL2_mixer -lGL -lglut -lGLU #-lSDL2_test
    message("Configured for 64bits GCC Linux")
}
linux-g++:contains(QMAKE_HOST.arch, x86):\
{
    # include SDL
    LIBS += -L/usr/lib -lSDL2 -lSDL2_image -lSDL2_ttf -lSDL2_gfx -lSDL2_mixer -lSDL2_test -lGL -lglut -lGLU
    message("Configured for 32bits GCC Linux")
}


linux-clang:contains(QMAKE_HOST.arch, x86_64):\
{
    # include SDL
    LIBS += -L/usr/lib64 -lSDL #-lSDL2_image -lSDL2_ttf -lSDL2_gfx -lSDL2_mixer -lGL -lglut -lGLU #-lSDL2_test
    message("Configured for 64bits CLANG Linux")
}
linux-clang:contains(QMAKE_HOST.arch, x86):\
{
    # include SDL
    LIBS += -L/usr/lib -lSDL #-lSDL2_image -lSDL2_ttf -lSDL2_gfx -lSDL2_mixer -lSDL2_test -lGL -lglut -lGLU
    message("Configured for 32bits CLANG Linux")
}
