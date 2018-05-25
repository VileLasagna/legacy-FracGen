
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

#include <mpi.h>
#include <png.h>



struct region{long double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)
bool operator==(const region& r1, const region& r2){return ( (r1.Imax - r2.Imax <= LDBL_EPSILON) && (r1.Imin - r2.Imin <= LDBL_EPSILON)
														  && (r1.Rmax - r2.Rmax <= LDBL_EPSILON) && (r1.Rmin - r2.Rmin <= LDBL_EPSILON) );}

region reg;
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

int myrank = 0;
int nprocs = 0;
size_t numDivs = 3;

png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers;

//std::array<std::future<bool>, numDivs> tasks;
std::vector<std::future<bool>> tasks(numDivs);







unsigned int getColour(unsigned int it/*, double x*/) noexcept
{
	

	if (ColourScheme)
	{
        //GO GO DUMB MATHS!!
		//Aproximate range: From 0.3 to 1018 and then infinity (O.o)
//		long double index = it + (log(2*(log(Bailout))) - (log(log(std::abs(x)))))/log(power);
//		return SDL_MapRGB(frac->format, (sin(index))*255, (sin(index+50))*255, (sin(index+100))*255);
        return SDL_MapRGB(frac->format, 128+ sin((float)it + 1)*128, 128 + sin((float)it)*128 ,  cos((float)it+1.5)*255);
	}
	else
	{
        //std::cout<< it <<std::endl;

        //return SDL_MapRGB(frac->format, 128+ sin((float)it + 1)*128, 128 + sin((float)it)*128 ,  cos((float)it+1.5)*255);
        if(it == max_iteration)
        {
            return SDL_MapRGB(frac->format, 0, 0 , 0);
        }
        else
        {
			//auto b = std::min(it,255u);
            return SDL_MapRGB(frac->format, std::min(it,255u) , 0, std::min(it,255u) );
            //return SDL_MapRGB(frac->format, 128+ sin((float)it + 1)*128, 128 + sin((float)it)*128 ,  cos((float)it+1.5)*255);
        }
	}

}


auto fracGen = [](region r,int index, Uint32* pix) noexcept
{
    //std::cout << "tid: " << std::this_thread::get_id() << std::endl;

	//Uint32* pix = (Uint32*)frac->pixels;
	long double incX = std::abs((r.Rmax - r.Rmin)/frac->w);
	long double incY = std::abs((r.Imax - r.Imin)/frac->h);
    for(int i = 0;i < height; i++)
	{
		if (i == frac->h)
		{
			return true;
		}
        //Initially intuitive/illustrative division
        //for(int j = (index%numDivs)*(frac->w/numDivs); j< ((index%numDivs)+1)*(frac->w/numDivs); j++)
        //Newer prefetcher-friendly version
        for(int j = 0 + index; j< frac->w; j+=numDivs)
		{

			Uint8* p = (Uint8*)pix + (i * frac->pitch) + j*frac->format->BytesPerPixel;//Set initial pixel
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

			*(Uint32*) p = getColour(iteration/*, x*/);
		}
	}
	return false;
};

void spawnTasks(region reg, bool bench) noexcept
{

    for(unsigned int i = 0; i < tasks.size(); i++)
	{
        tasks[i] = std::async(std::launch::async, fracGen,reg, i, /*tasks.size(),*/ (Uint32*)frac->pixels);
	}


    for(unsigned int i = 0; i < tasks.size(); i++)
	{
		if(tasks[i].get())
		{
			auto d = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();
		}
	}

}


int runProgram(bool benching) noexcept
{
	endProgram = false;
	SDL_Init(SDL_INIT_EVERYTHING);
//	screen = SDL_SetVideoMode(w,h,bpp, SDL_HWSURFACE|SDL_DOUBLEBUF|SDL_ASYNCBLIT);
	mainwindow = SDL_CreateWindow("Mandelbrot Fractal Explorer - Use Mouse1 to zoom in and Mouse2 to zoom out. Press Mouse 3 to change colouring scheme",
							  SDL_WINDOWPOS_UNDEFINED,
							  SDL_WINDOWPOS_UNDEFINED,
							  w, h,
							  SDL_WINDOW_SHOWN);

	render = SDL_CreateRenderer(mainwindow, -1, 0);

	screen = SDL_CreateRGBSurface(0, w, h, bpp,
									0x00FF0000,
									0x0000FF00,
									0x000000FF,
									0xFF000000);

	texture = SDL_CreateTexture(render,
								SDL_PIXELFORMAT_ARGB8888,
								SDL_TEXTUREACCESS_STREAMING,
								w, h);
	assert(screen);
//	SDL_WM_SetCaption("Mandelbrot Fractal Explorer - Use Mouse1 to zoom in and Mouse2 to zoom out. Press Mouse 3 to change colouring scheme",0);
	SDL_SetWindowTitle(mainwindow, "Mandelbrot Fractal Explorer - Use Mouse1 to zoom in and Mouse2 to zoom out. Press Mouse 3 to change colouring scheme");

	//frac =	SDL_CreateRGBSurface(SDL_HWSURFACE|SDL_DOUBLEBUF|SDL_ASYNCBLIT,w,h,bpp,0,0,0,0);
	frac = SDL_CreateRGBSurface(0, w, h, bpp,
								0x00FF0000,
								0x0000FF00,
								0x000000FF,
								0xFF000000);

	
	reg.Imax = 1.5;
	reg.Imin = -1.5;
	reg.Rmax = 1;
	reg.Rmin = -2;


	//CreateFractal(reg);
    genTime = std::chrono::high_resolution_clock::now();
    spawnTasks(reg, benching);
	createHighlight();

	while(!endProgram)
	{
		if(creating)
		{
            spawnTasks(reg, benching);
		}

        if(!benching)
        {
            capture();
        }
		paint();
	}

	return 0;
}

void printUsage()
{
    std::vector<std::string> help
    {
        "Fracgen is a toy mandelbrot fractal generator you can use for silly CPU benchmarks",
        "If you just want to look at some fractals, just run it plain",
        "Drag boxes with Mouse1 to select region of interest, Mouse2 switches colour scheme",
        "Mouse 3 resets the image to the original area",
        "",
        "Run from the cli for toy benchmarking",
        "Available options",
        "    -i X",
        "        Number of interactions to run",
        "    -j X",
        "        Number of parallel tasks to run",
        "    -o X",
        "        Output results to a file"

    };
    for(std::string h: help)
    {
        std::cout << h << std::endl;
    }
}

int main (int argc, char** argv) noexcept
{

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    int res = 0;
    std::ofstream outlog;
    auto numThreads = std::thread::hardware_concurrency();
    if(numThreads > 0)
    {
        // This 4 is an experimental factor. It feels like a sweet spot
        // I don't know why and never chased this but it's been true
        // across Bulldozer, Skylake and Threadripper
        numDivs=numThreads*4;
        tasks.resize(numDivs);
    }


    res = runProgram(false);

    if(outlog.is_open())
    {
        outlog.flush();
        outlog.close();
    }

    MPI_Finalize();

    return res;
}

