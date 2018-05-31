#include "FracGen.hpp"

#include <cuda_runtime.h>
#include <iostream>

__device__ uint32_t getColour(unsigned int it)
{
    RGBA colour;
    colour.g = min(it, 255u);
    return *(reinterpret_cast<uint32_t*>(&colour));

}

__global__ void calculateIterations(uint32_t* data, int width, int height, Region r)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    int index = (row*width)+col;
    if (index > width*height)
    {
        return;
    }
    unsigned int iteration_factor = 100;
    unsigned int max_iteration = 256 * iteration_factor;

    double incX = r.Rmax - r.Rmin/width;
    double incY = r.Imax - r.Imin/height;

    double x = r.Rmin+(col*incX);
    double y = r.Imax-(row*incY);
    double x0 = x;
    double y0 = y;

    unsigned int iteration = 0;

    while ( (x*x + y*y <= 4)  &&  (iteration < max_iteration) )
    {
        double xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;

        x = xtemp;

        iteration++;
    }
    data[index] = getColour(iteration);

}

void FracGen::Generate(ImageData v, int channels, int width, int height, Region r)
{
    uint32_t* devVect;
    cudaMallocManaged(&devVect, width*height*sizeof(uint32_t));
    calculateIterations<<<width,height>>>(devVect, width, height, r);
    cudaDeviceSynchronize();
    //std::cout << "Final colour data: ";
    uint8_t* colourVect = reinterpret_cast<uint8_t*>(devVect);
    for(size_t i = 0; i < width*height * channels; i++)
    {
        v[i] = colourVect[i];
        //std::cout << colourVect[i] << " ";
    }
    //std::cout << std::endl;
    cudaFree(devVect);
}

FracGen::FracGen()
{

}

FracGen::~FracGen()
{
    cudaDeviceReset();
}
