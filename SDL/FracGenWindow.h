#ifndef TOYBROT_FRACGENWINDOW_DEFINED
#define TOYBROT_FRACGENWINDOW_DEFINED

#ifdef WIN32
    #include <Windows.h>
    #include "SDL.h"
    #include "SDL_opengl.h"
#else

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#endif

#include <memory>
#include <atomic>
#include <cfloat>

struct Region{long double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)

bool operator==(const Region& r1, const Region& r2);

using SurfPtr = std::shared_ptr<SDL_Surface>;
using SurfUnq = std::unique_ptr<SDL_Surface>;


class FracGenWindow
{
public:
    FracGenWindow(int width, int height, int bpp, std::shared_ptr<std::atomic_bool> redrawFlag);
    ~FracGenWindow();

    void draw(std::shared_ptr<SDL_Surface> surface);
    void paint();
    float AspectRatio() const {return height > 0 ? width/height : 0;}
    bool captureEvents();
    void registerRedrawFlag(std::shared_ptr<std::atomic_bool> b) {redrawRequired = b;}
    void registerColourFlag(std::shared_ptr<int> i) {colourScheme = i;}
    void setRegion(std::shared_ptr<Region> r) {ROI = r;}
    SurfPtr getFrac() {return frac;}
    void setTitle(std::string title);


private:

    void drawHighlight();

    bool onKeyboardEvent(const SDL_KeyboardEvent& e) noexcept;
    bool onMouseMotionEvent(const SDL_MouseMotionEvent& e) noexcept;
    bool onMouseButtonEvent(const SDL_MouseButtonEvent& e ) noexcept;

    int  width;
    int  height;
    int  colourDepth;
    bool    drawRect;
    int     rectX;
    int     rectY;
    int     mouseX;
    int     mouseY;

    SDL_Window* mainwindow;
    SDL_Renderer* render;
    SurfPtr screen;
    SDL_Texture* texture;
    SurfPtr frac;
    SurfUnq highlight;

    std::shared_ptr<std::atomic_bool> redrawRequired;
    std::shared_ptr<Region> ROI;
    std::shared_ptr<int> colourScheme;
};

#endif //TOYBROT_FRACGENWINDOW_DEFINED
