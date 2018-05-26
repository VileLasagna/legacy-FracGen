
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
#include <png.h>



struct region{long double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)
bool operator==(const region& r1, const region& r2){return ( (r1.Imax - r2.Imax <= LDBL_EPSILON) && (r1.Imin - r2.Imin <= LDBL_EPSILON)
														  && (r1.Rmax - r2.Rmax <= LDBL_EPSILON) && (r1.Rmin - r2.Rmin <= LDBL_EPSILON) );}

region reg, myReg;
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
std::stringstream stream;

int myrank = 1;
int nprocs = 1;
size_t numDivs = 3;
int realDivs = 0;
int imDivs = 0;

using pngColourType = png_byte;

pngColourType color_type;
png_byte bit_depth;
png_bytep *row_pointers;

png_structp pngWritePtr = nullptr;
png_infop pngInfoPtr = nullptr;
png_bytep row = nullptr;
//std::array<std::future<bool>, numDivs> tasks;
std::vector<std::future<bool>> tasks(numDivs);

FILE* fp;


struct pngRGB
{
    pngColourType r;
    pngColourType g;
    pngColourType b;
};

std::vector<std::vector<pngRGB> > pngRows;


pngRGB getColour(unsigned int it, unsigned int div) noexcept
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
            switch (div % 3)
            {
                case 0: colour.r = 0; break;
                case 1: colour.g = 0; break;
                case 2: colour.b = 0; break;
            }
        }
	}
    return colour;
}


auto fracGen = [](region r,int index, int numTasks, std::vector<std::vector<pngRGB> >* rows) noexcept
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

            rows->at(i)[j] = getColour(iteration, index);
		}
	}
	return false;
};

void spawnTasks(region reg, bool bench) noexcept
{

    for(unsigned int i = 0; i < tasks.size(); i++)
	{
        tasks[i] = std::async(std::launch::async, fracGen,reg, i, tasks.size(), &pngRows);
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
	
	reg.Imax = 1.5;
	reg.Imin = -1.5;
	reg.Rmax = 1;
	reg.Rmin = -2;


	//CreateFractal(reg);
    genTime = std::chrono::high_resolution_clock::now();
    spawnTasks(reg, benching);


    spawnTasks(reg, benching);

	return 0;
}

void allocRows()
{
    pngRows.resize(height);
    for(auto& v: pngRows)
    {
        v.resize(width);
    }
}

int initPNG(int rank, int procs)
{
    //std::string filename("/mnt/pandora/storage/users/jehferson/FracGenOut/FracGenMPI");
    std::string filename("FracGenOut/FracGenMPI");
    filename.append(std::to_string(myrank));
    filename.append(".png");

    fp = fopen(filename.c_str(), "wb");
    if (fp == NULL)
    {
        std::cerr << "Could not open file " << filename << " for writing" << std::endl;
        return 1;
    }


    // Initialize PNG write structure
    pngWritePtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (pngWritePtr == nullptr)
    {
        std::cerr << "Could not allocate PNG write struct" << std::endl;
        return 2;
    }

    // Initialize info structure
    pngInfoPtr = png_create_info_struct(pngWritePtr);
    if (pngInfoPtr == nullptr)
    {
        std::cerr << "Could not allocate PNG info struct" << std::endl;
        return 2;
    }

    png_init_io(pngWritePtr, fp);

    // Write header (8 bit colour depth)
    png_set_IHDR(pngWritePtr, pngInfoPtr, width, height,
    8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);


    std::string title = "FracGenMPI Mandelbrot section ";
    title.append(std::to_string(rank));
    title.append(" of ");
    title.append(std::to_string(procs));
    char ctitle[256];
    for(char& c: ctitle)
    {
        c = 0;
    }
    title.copy(ctitle,title.length(),0);

    png_text title_text;
    title_text.compression = PNG_TEXT_COMPRESSION_NONE;
    title_text.key = "Title";
    title_text.text = ctitle;
    title_text.text_length = title.size();
    png_set_text(pngWritePtr, pngInfoPtr, &title_text, 1);


    png_write_info(pngWritePtr, pngInfoPtr);

    allocRows();

    return 0;
}

int writePNG()
{
    int res = 0;
    // Write image data
    int x, y;
    for (auto row : pngRows)
    {
       png_write_row(pngWritePtr, reinterpret_cast<png_const_bytep>(row.data()) );
    }

    // End write
    png_write_end(pngWritePtr, NULL);
    return res;
}

int main (int argc, char** argv) noexcept
{

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if(nprocs == 0 )
    {
        myrank = 1;
        nprocs = 1;
    }

    int res = 0;
    std::ofstream outlog;

    allocRows();
    res = initPNG(myrank,nprocs);

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
        res = writePNG();
    }

    MPI_Finalize();

    return res;
}

