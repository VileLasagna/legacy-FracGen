
CONFIG += app c++14
CONFIG -= qt

CONFIG(release, debug|release) \
{
    DESTDIR = $$_PRO_FILE_PWD_/distro/release
}


CONFIG(debug, debug|release) \
{
    DESTDIR = $$_PRO_FILE_PWD_/distro/debug
}

CLANG = $$find(QMAKESPEC, clang* )
!equals(CLANG, "") :\
{
    QMAKE_CXXFLAGS_WARN_ON = -Wall -Wno-ignored-qualifiers -Wno-missing-braces -Wno-unknown-pragmas
    QMAKE_CXXFLAGS_WARN_ON += -Wno-macro-redefined -Wno-microsoft-exception-spec -Wno-missing-variable-declarations
}

SOURCES += \
    FracGen/main.cpp

#QMAKE_LFLAGS += /NODEFAULTLIB:msvcrt.lib

#linux-g++:contains(QMAKE_HOST.arch, x86_64):\
#{
#    # include SDL
#    LIBS += -L/usr/lib64 -lSDL
#    !build_pass:message("Configured for 64bits GCC Linux")
#}
#linux-g++:contains(QMAKE_HOST.arch, x86):\
#{
#    # include SDL
#    LIBS += -L/usr/lib -lSDL
#    !build_pass:message("Configured for 32bits GCC Linux")
#}


#linux-clang:contains(QMAKE_HOST.arch, x86_64):\
#{
#    # include SDL
#    LIBS += -L/usr/lib64 -lSDL
#    !build_pass:message("Configured for 64bits CLANG Linux")
#}
#linux-clang:contains(QMAKE_HOST.arch, x86):\
#{
#    # include SDL
#    LIBS += -L/usr/lib -lSDL
#    !build_pass:message("Configured for 32bits CLANG Linux")
#}

win32:\
{
    SDL_PATH = "C:/Work/libs/SDL2-2.0.5/VC/"
    VC_LIBS_PATH = "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC"


#    LIBS += \
#        -L"C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\lib\\amd64"                               \
#        "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\lib\\amd64\\legacy_stdio_definitions.lib"   \
#        -L$$PWD/../../../libs/SDL2-2.0.5/VC/lib/x64 "C:\\Work\\libs\\SDL2-2.0.5\\VC\\lib\\x64\\SDL2.lib"        \
#        "C:\\Work\\libs\\SDL2-2.0.5\\VC\\lib\\x64\\SDL2main.lib"

    LIBS += \
            -L$$SDL_PATH/lib/x64        \
            -L$$VC_LIBS_PATH/lib/amd64  \
            -llegacy_stdio_definitions -lSDL2 -lSDL2main

    INCLUDEPATH += \
                $$SDL_PATH/include      \
                $$VC_LIBS_PATH/include

    #DEPENDPATH += "C:\\Work\\libs\\SDL2-2.0.5\\VC\\bin" "C:\\Work\\libs\\SDL2-2.0.5\\VC\\include"
    !build_pass:message("Configured for windows")

}
else:\
{
    LIBS += -lSDL2 -lSDL2main -lpthread
}


#win32: LIBS += -L$$PWD/../../../libs/SDL-1.2.15/lib/ -llibSDL.dll

#INCLUDEPATH += $$PWD/../../../libs/SDL-1.2.15/include
#DEPENDPATH += $$PWD/../../../libs/SDL-1.2.15/include
