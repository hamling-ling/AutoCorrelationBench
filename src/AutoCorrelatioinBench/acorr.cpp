#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__)
// place cl.hpp to local directory for Mac OS
#include "cl.hpp"
#else
#include <CL/cl.hpp>
#endif

#include "util.hpp"
#include "err_code.h"
#include "device_picker.hpp"

//------------------------------------------------------------------------------
//  Constants
//------------------------------------------------------------------------------
#if defined(__APPLE__)
// my personal debug setting. need to change later
#define CL_PATH         "/Users/nobu/GitHub/AutoCorrelationBench/src/AutoCorrelatioinBench/acorr.cl"
#define DEVICE_ID       1
#else
#define CL_PATH         "acorr.cl"
#define DEVICE_ID       0
#endif
#define SAMPLE_SIZE_N   1024      // Sample Size
#define WORK_GROUP_SIZE 128       // Workgroup size

//------------------------------------------------------------------------------
//  functions
//------------------------------------------------------------------------------
extern double wtime();   // Returns time since some fixed past point (wtime.c)

using namespace std;

int main(int argc, char *argv[])
{
    int         N          = SAMPLE_SIZE_N;  // Real data sampling size
    util::Timer timer;                       // timing

    std::vector<float> h_sample( N, 0);   // N complex samples as an input

	for ( int i = 0; i < h_sample.size(); i++) {
		h_sample[i]   = (float)sin(3.5 * i * M_PI / N);
        h_sample[i+1] = 0.0f;
        //cout << "orig[" << i << "] = " << h_sample[i] << endl;
	}

    try
    {
        cl_uint deviceIndex = DEVICE_ID;

        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList( devices);

        // Check device index in range
        if ( deviceIndex >= numDevices)
        {
            std::cout << "Invalid device index\n";
            return EXIT_FAILURE;
        }

        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName( device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back( device);

        cl::Context      context( chosen_device);
        cl::CommandQueue queue( context, device);

        // Create the compute program from the source buffer
        cl::Program program = cl::Program(context, util::loadProgram(CL_PATH), true);

        // Create the compute kernel from the program
        cl::NDRange global(N);
        // Setup device global memory
        cl::Buffer d_sample = cl::Buffer(context, h_sample.begin(), h_sample.end(), true);
        cl::Buffer d_output = cl::Buffer(context,
                                         CL_MEM_READ_WRITE,
                                         sizeof(h_sample[0]) * h_sample.size()
                                         );

        // bench mark for basic kernel
        {
            cl::make_kernel<int, cl::Buffer, cl::Buffer> acorr( program, "acorr");
            double start_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0;
            for(int i = 0; i < 10000; i++) {
                // compute
                acorr(cl::EnqueueArgs( queue, global),
                      N,
                      d_sample,
                      d_output
                      );
                queue.finish();
            }
            double run_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0 - start_time;

            std::vector<float> h_output( N, 0);
            cl::copy(queue, d_output, h_output.begin(), h_output.end());
            cout << run_time << " sec" << endl;
        }

        // bench mark using local mem
        {
            cl::NDRange local(WORK_GROUP_SIZE);
            cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg> acorr_local( program, "acorr_local");
            cl::LocalSpaceArg localmem = cl::Local(sizeof(float) * WORK_GROUP_SIZE);

            double start_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0;
            for(int i = 0; i < 10000; i++) {
                // compute
                acorr_local(cl::EnqueueArgs( queue, global, local),
                      N,
                      d_sample,
                      d_output,
                      localmem
                      );
                queue.finish();
            }
            double run_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0 - start_time;

            std::vector<float> h_output( N, 0);
            cl::copy(queue, d_output, h_output.begin(), h_output.end());
            cout << run_time << " sec" << endl;
        }
    } catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
