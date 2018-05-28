#ifndef PNGWRITER_H_DEFINED
#define PNGWRITER_H_DEFINED
#include <png.h>
#include <vector>

using pngColourType = png_byte;

struct pngRGB
{
    pngColourType r;
    pngColourType g;
    pngColourType b;
};

using pngData = std::vector<std::vector<pngRGB>>;

class pngWriter
{
    public:

        pngWriter(unsigned int width = 0, unsigned int height = 0);

        void setWidth(unsigned int width)   noexcept {w = width;}
        void setHeight(unsigned int height) noexcept {h = height;}

        bool Init();
        bool Write(const pngData& rows);
        void Alloc(pngData& rows);

    private:

        unsigned int w, h;

        png_byte bit_depth;
        png_structp writePtr;
        png_infop infoPtr;

        FILE* output;

};

#endif //PNGWRITER_H_DEFINED
