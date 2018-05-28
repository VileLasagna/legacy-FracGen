
#define NOMINMAX


#include <algorithm>
#include <assert.h>
#include <sstream>
#include <array>
#include <future>
#include <chrono>
#include <cfloat>
#include <numeric>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <mpi.h>
#include "pngWriter.h"



struct region{long double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)
bool operator==(const region& r1, const region& r2){return ( (r1.Imax - r2.Imax <= LDBL_EPSILON) && (r1.Imin - r2.Imin <= LDBL_EPSILON)
														  && (r1.Rmax - r2.Rmax <= LDBL_EPSILON) && (r1.Rmin - r2.Rmin <= LDBL_EPSILON) );}

region reg {-1.5,1.5,-2,2};
region myReg {reg};
bool endProgram;
unsigned int iteration_factor = 100;
unsigned int max_iteration = 256 * iteration_factor;
long double Bailout = 2;
long double power = 2;
int width =	1280;
int height = 720;
int lastI = 0;
bool ColourScheme = false;
auto genTime (std::chrono::high_resolution_clock::now());

int myrank = 1;
int nprocs = 1;
size_t numDivs = 3;
int realDivs = 0;
int imDivs = 0;



png_bytep row = nullptr;
//std::array<std::future<bool>, numDivs> tasks;
std::vector<std::future<bool>> tasks(numDivs);



std::vector<std::vector<pngRGB> > pngRows;


pngRGB getColour(unsigned int it, unsigned int rank) noexcept
{
    pngRGB colour;

	if (ColourScheme)
	{
        colour.r = 128 + std::sin((float)it + 1)*128;
        colour.g = 128 + std::sin((float)it)*128;
        colour.b = std::cos((float)it+1.5)*255;
	}
	else
	{
        if(it == max_iteration)
        {
            colour.r = 0;
            colour.g = 0;
            colour.b = 0;
        }
        else
        {
            colour.r = std::min(it,255u);
            colour.g = std::min(it,255u);
            colour.b = std::min(it,255u);

            //let's make it a bit fun
            switch (rank % 7)
            {
                case 0: colour.r = 0;                             break;
                case 1:               colour.g = 0;               break;
                case 2:                             colour.b = 0; break;
                case 3:               colour.g = 0; colour.b = 0; break;
                case 4: colour.r = 0;               colour.b = 0; break;
                case 5: colour.r = 0; colour.g = 0;               break;
                case 6: break;
            }
        }
	}
    return colour;
}


auto fracGen = [](region r,int index, int numTasks, int rank, std::vector<std::vector<pngRGB> >* rows) noexcept
{
    if (rows == nullptr)
    {
        return false;
    }
    size_t pixHeight = rows->size();
    size_t pixWidth = rows->at(0).size();

    long double incX = std::abs((r.Rmax - r.Rmin)/pixWidth);
    long double incY = std::abs((r.Imax - r.Imin)/pixHeight);

    for(int i = index * pixHeight/numTasks; i < (index + 1) * pixHeight/numTasks; i++)
	{
        if (i == rows->size())
		{
			return true;
		}

        for(int j = 0; j < pixWidth; j++)
		{

			long double x = r.Rmin+(j*incX);
			long double y = r.Imax-(i*incY);
			long double x0 = x;
			long double y0 = y;

			unsigned int iteration = 0;

			while ( (x*x + y*y <= 4)  &&  (iteration < max_iteration) )
			{
				long double xtemp = x*x - y*y + x0;
				y = 2*x*y + y0;

				x = xtemp;

				iteration++;
			}

            rows->at(i)[j] = getColour(iteration, rank);
		}
	}
	return false;
};

void spawnTasks(region reg, bool bench) noexcept
{
    std::cout << "Task " << myrank << " of " << nprocs << " drawing region: ";
    std::cout << myReg.Imin << "i -> " << myReg.Imax << "i // " << myReg.Rmin << " -> " << myReg.Rmax << std::endl;

    for(unsigned int i = 0; i < tasks.size(); i++)
	{
        tasks[i] = std::async(std::launch::async, fracGen,reg, i, tasks.size(), myrank, &pngRows);
	}

    for(unsigned int i = 0; i < tasks.size(); i++)
	{
        //block until all tasks are done
		if(tasks[i].get())
		{
			auto d = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();
		}
	}

}


int runProgram(bool benching) noexcept
{
	//CreateFractal(reg);
    genTime = std::chrono::high_resolution_clock::now();
    spawnTasks(myReg, benching);

	return 0;
}

void defineRegion(unsigned int rank, unsigned int procs)
{
    region cropped;
    //find the closest square
    int p = procs;
    int r = sqrt(p);
    while( (r*r) != p )
    {
        p -= 1;
        r = sqrt(p);
    }
    int left = procs % p;
    if(rank == 1)
    {
        std::cout << "From " << procs << " tasks we have a " << r << "x" << r << " + " << left << " area " << std::endl;
    }
    //we have R*R + left processes;
    long double imStep = (reg.Imax - reg.Imin) / r;
    long double imStepover = 0;
    long double rStep = (reg.Rmax - reg.Rmin) / r;
    if(left != 0)
    {
        rStep = (reg.Rmax - reg.Rmin) / (r + 1);
        imStepover = (reg.Imax - reg.Imin) / left;
    }

    if(rank < r*r)
    {
        myReg.Imax = reg.Imax - (imStep*(rank/r));
        myReg.Imin = myReg.Imax - imStep;

        myReg.Rmin = reg.Rmin + (rStep* (rank%r));
        myReg.Rmax = myReg.Rmin + rStep;
    }
    else
    {
        myReg.Imax = reg.Imax - (imStepover*(rank%r));
        myReg.Imin = myReg.Imax - imStepover;

        myReg.Rmin = reg.Rmin + (rStep * r);
        myReg.Rmax = myReg.Rmin + rStep;

        height = height * imStepover / imStep;

    }

}

int main (int argc, char** argv) noexcept
{

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if(nprocs == 0 )
    {
        myrank = 0;
        nprocs = 1;
    }

    pngWriter writer(4096,1920);
    writer.Init();
    writer.Alloc(pngRows);
    defineRegion(myrank,nprocs);


    int res = 0;
    std::ofstream outlog;


    if( res == 0)
    {
        res = runProgram(false);
    }

    if(outlog.is_open())
    {
        outlog.flush();
        outlog.close();
    }

    if(res == 0)
    {
        writer.Write(pngRows);
    }

    MPI_Finalize();

    return res;
}

