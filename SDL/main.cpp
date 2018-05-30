
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

#include "FracGenWindow.h"



unsigned int iteration_factor = 100;
unsigned int max_iteration = 256 * iteration_factor;
long double Bailout = 2;
long double power = 2;

std::shared_ptr<int> ColourScheme(new int(0));
auto genTime (std::chrono::high_resolution_clock::now());

size_t numDivs = std::thread::hardware_concurrency() * 4;

//std::array<std::future<bool>, numDivs> tasks;
std::vector<std::future<bool>> tasks(numDivs);


uint32_t getColour(unsigned int it, SurfPtr frac) noexcept
{
	

    if ((*ColourScheme)%2)
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

auto fracGen = [](Region r,int index, SurfPtr frac, int h0) noexcept
{

	long double incX = std::abs((r.Rmax - r.Rmin)/frac->w);
	long double incY = std::abs((r.Imax - r.Imin)/frac->h);
	for(int i = h0;i < h0+10; i++)
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

            uint8_t* p = reinterpret_cast<uint8_t*>(frac->pixels) + (i * frac->pitch) + j*frac->format->BytesPerPixel;//Set initial pixel
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

            *(uint32_t*) p = getColour(iteration, frac/*, x*/);
		}
	}
	return false;
};

bool spawnTasks(Region reg, bool bench, std::shared_ptr<SDL_Surface> frac) noexcept
{
	static std::atomic<int> h {0};

    SDL_LockSurface(frac.get());
    for(unsigned int i = 0; i < tasks.size(); i++)
	{
        tasks[i] = std::async(std::launch::async, fracGen,reg, i, frac, h.load());
	}

    h+= 10;

    bool finishedGeneration = false;
    for(unsigned int i = 0; i < tasks.size(); i++)
	{
		if(tasks[i].get())
		{
			h.store(0);
            finishedGeneration = true;
		}
	}
    SDL_UnlockSurface(frac.get());

    return finishedGeneration;

}




int runProgram(bool benching) noexcept
{
    bool keepRunning = true;


    std::shared_ptr<Region> reg(new Region());
	
    reg->Imax = 1.5;
    reg->Imin = -1.5;
    reg->Rmax = 1;
    reg->Rmin = -2;

    std::shared_ptr<std::atomic_bool> redraw(new std::atomic_bool);
    redraw->store(true);

    FracGenWindow mainWindow(1280,1024,32, redraw);
    mainWindow.registerColourFlag(ColourScheme);
    mainWindow.setRegion(reg);

    SurfPtr frac = mainWindow.getFrac();
    std::stringstream stream;


    genTime = std::chrono::high_resolution_clock::now();
    spawnTasks(*reg, benching, frac);
    bool clockReset = false;

    while(keepRunning)
	{
        if(redraw->load())
		{
            if(clockReset)
            {
                clockReset = false;
                genTime = std::chrono::high_resolution_clock::now();
            }
            bool b = spawnTasks(*reg, benching, frac);
            if (b)
            {
                redraw->store(false);
                auto d = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();

                stream << "Fractal Generation Took " << d << " milliseconds";
                mainWindow.setTitle(stream.str());
                stream.str("");
                clockReset = true;
            }

		}

        if(!benching)
        {
            keepRunning = mainWindow.captureEvents();
        }
        mainWindow.paint();
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
    int res = 0;
    size_t iterations = 1;
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

    enum class setting{NONE, ITERATIONS, JOBS, OUTPUT};

    if(argc > 1)
    {
        auto op = setting::NONE;
        for(int a = 1; a < argc; a++)
        {
            std::string token(argv[a]);
            if(token == "-i")
            {
                    op = setting::ITERATIONS;
                    continue;
            }
            if(token == "-j")
            {
                    op = setting::JOBS;
                    continue;
            }
            if(token == "-o")
            {
                    op = setting::OUTPUT;
                    continue;
            }
            if((token == "-h") || (token == "--h"))
            {
                printUsage();
                return 0;
            }

            //No exceptions here, only undefined behaviour
            int n = atoi(argv[a]);
            switch(op)
            {
                case setting::ITERATIONS:
                    iterations = n;
                    op = setting::NONE;
                    break;
                case setting::JOBS:
                    numDivs = n;
                    tasks.resize(numDivs);
                    op = setting::NONE;
                    break;
                case setting::OUTPUT:
                    outlog.open(argv[a]);
                    if(outlog.fail())
                    {
                        std::cout << "Could not open file " << argv[a] << " for output";
                        return 1;
                    }
                    op = setting::NONE;
                    break;
                default:
                    break;
            }


        }

        std::cout << "Preparing to run with "<< numDivs << " parallel tasks" << std::endl;
        if(outlog.is_open())
        {
            outlog << "Preparing to run with "<< numDivs << " parallel tasks" << std::endl;
        }
    }
    if(iterations > 1)
    {
        std::vector<size_t> results;
        for(size_t i = 0; i < iterations; i++)
        {
            res = runProgram(true);
            if(res != 0)
            {
                return res;
            }
            auto d = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();
            std::cout << "Iteration " << i << " took " << d << " milliseconds" << std::endl;
            if(outlog.is_open())
            {
                outlog << "Iteration " << i << " took " << d << " milliseconds" << std::endl;
            }
            results.push_back(d);
        }
        auto avg = std::accumulate(results.begin(), results.end(), 0)/ results.size();

        std::cout << std::endl << "Average time of " << avg << " milliseconds (over " << results.size()<< " tests)"<< std::endl;
        if(outlog.is_open())
        {
            outlog << std::endl << "Average time of " << avg << " milliseconds (over " << results.size()<< " tests)"<< std::endl;
        }
    }
    else
    {
        res = runProgram(false);
    }

    if(outlog.is_open())
    {
        outlog.flush();
        outlog.close();
    }

    return res;
}

