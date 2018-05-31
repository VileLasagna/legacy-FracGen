#include "FracGen.hpp"

#include <cuda_runtime.h>
#include <iostream>

__global__ void calculateIterations(int* data, int width, int height, Region r)
{
    unsigned int iteration_factor = 100;
    unsigned int max_iteration = 256 * iteration_factor;

    double incX = r.Rmax - r.Rmin/width;
    double incY = r.Imax - r.Imin/height;
    for(int i = 0;i < height; i++)
    {
        for(int j = 0; j< width; j++)
        {

            double x = r.Rmin+(j*incX);
            double y = r.Imax-(i*incY);
            double x0 = x;
            double y0 = y;

            unsigned int iteration = 0;

//            while ( (x*x + y*y <= 4)  &&  (iteration < max_iteration) )
//            {
//                double xtemp = x*x - y*y + x0;
//                y = 2*x*y + y0;

//                x = xtemp;

//                iteration++;
//            }
            data[(i*width)+j] = iteration;
            data[(i*width)+j] = 128;

        }
    }
}


void FracGen::Iterations(std::vector<int>& v, int width, int height, Region r)
{
    int* devVect;
    cudaMalloc(&devVect, v.size()*sizeof(int));

    cudaMemcpy(devVect, v.data(), v.size()*sizeof(int), cudaMemcpyHostToDevice);

    calculateIterations<<<1,1>>>(devVect, width, height, r);

    cudaDeviceSynchronize();
    cudaMemcpy(devVect, v.data(), v.size()*sizeof(int), cudaMemcpyDeviceToHost);

}

void FracGen::Generate(ImageData v, int channels, int width, int height, Region r)
{
    std::vector<int> it(width*height);
    Iterations(it, width, height, r);
    size_t imgIdx = 0;
    std::cout <<" Iteration values: ";
    for(size_t idx = 0; idx < it.size(); idx++)
    {
        std::cout<< it[idx] << " ";
        for(size_t colour = 0; colour < channels; colour++)
        {
            //green or bust
            v[imgIdx] = colour%channels == 1 ? static_cast<colourType>(std::min(it[idx],255)) : 0;
            imgIdx++;
        }
    }
    std::cout << std::endl;
}

FracGen::FracGen()
{

}

FracGen::~FracGen()
{
    cudaDeviceReset();
}
