#ifndef FRACGEN_HPP_DEFINED
#define FRACGEN_HPP_DEFINED

#include <vector>
#include <cstdint>



struct Region{long double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)

bool operator==(const Region& r1, const Region& r2);

using colourType = uint8_t;
using ImageData = colourType*;

class FracGen
{
public:

    FracGen();
    ~FracGen();

    void Generate(ImageData img, int channels, int width, int height, Region r);

private:

    void Iterations(std::vector<int>& v, int width, int height, Region r);
    uint32_t getColour();
};

#endif //FRACGEN_HPP_DEFINED
