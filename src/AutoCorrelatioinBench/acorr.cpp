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
#define SAMPLE_SIZE_N   8      // Sample Size
#define WORK_GROUP_SIZE 4       // Workgroup size
#define LOOPS           1     // Iteration for benchmark
//------------------------------------------------------------------------------
//  functions
//------------------------------------------------------------------------------
extern double wtime();   // Returns time since some fixed past point (wtime.c)

using namespace std;

int main(int argc, char *argv[])
{
    int         N          = SAMPLE_SIZE_N;  // Real data sampling size
    util::Timer timer;                       // timing
    // N samples + some for vector computation as an input
    std::vector<float> h_sample( N + 32, 0);
    std::vector<uint16_t> h_sample16( N + 32, 0);
    std::vector<__fp16> h_samplefp16( N + 32, 0);

    for ( int i = 0; i < N; i++) {
        h_sample[i]   = (float)sin(3.5 * i * M_PI / N);
        h_sample16[i] = round(h_sample[i] /63665.0f);
        h_samplefp16[i] = (__fp16)h_sample[i];
        cout << "fp32[" << i << "] = " << h_sample[i] << endl;
        cout << "fp16[" << i << "] = " << h_samplefp16[i] << endl;
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
        cl::NDRange local(WORK_GROUP_SIZE);
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
            for(int i = 0; i < LOOPS; i++) {
                // compute
                acorr(cl::EnqueueArgs( queue, global, local),
                      N,
                      d_sample,
                      d_output
                      );
                queue.finish();
            }
            double run_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0 - start_time;

            std::vector<float> h_output( N, 0);
            cl::copy(queue, d_output, h_output.begin(), h_output.end());

            cout << "basic result ----" << endl;
            cout << run_time << " sec" << endl;
            for(int i = 0; i < N; i++) {
                //cout << "[" << i << "]=" << h_output[i] << endl;
            }
        }

        // bench mark using local mem
        {
            cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::LocalSpaceArg> acorr_local( program, "acorr_local");
            cl::LocalSpaceArg localmem = cl::Local(sizeof(float) * WORK_GROUP_SIZE);

            double start_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0;
            for(int i = 0; i < LOOPS; i++) {
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

            cout << "local mem result ----" << endl;
            cout << run_time << " sec" << endl;
        }

        // bench mark using foat4
        {
            cl::make_kernel<int, cl::Buffer, cl::Buffer> acorr_vec4( program, "acorr_vec4");

            double start_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0;
            for(int i = 0; i < LOOPS; i++) {
                // compute
                acorr_vec4(cl::EnqueueArgs( queue, global, local),
                      N,
                      d_sample,
                      d_output
                      );
                queue.finish();
            }
            double run_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0 - start_time;

            std::vector<float> h_output( N, 0);
            cl::copy(queue, d_output, h_output.begin(), h_output.end());

            cout << "float4 result ----" << endl;
            cout << run_time << " sec" << endl;
            for(int i = 0; i < N; i++) {
                //cout << "[" << i << "]=" << h_output[i] << endl;
            }
        }

        // bench mark using ushort16
        {
            cl::Buffer d_sample16 = cl::Buffer(context, h_sample16.begin(), h_sample16.end(), true);
            cl::Buffer d_output16 = cl::Buffer(context,
                                             CL_MEM_READ_WRITE,
                                             sizeof(h_sample16[0]) * h_sample16.size()
                                             );

            cl::make_kernel<int, cl::Buffer, cl::Buffer> acorr_us8( program, "acorr_us8");

            double start_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0;
            for(int i = 0; i < LOOPS; i++) {
                // compute
                acorr_us8(cl::EnqueueArgs( queue, global, local),
                      N,
                      d_sample16,
                      d_output16
                      );
                queue.finish();
            }
            double run_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0 - start_time;

            std::vector<uint16_t> h_output16( N, 0);
            cl::copy(queue, d_output, h_output16.begin(), h_output16.end());

            cout << "ushort8 result ----" << endl;
            cout << run_time << " sec" << endl;
            for(int i = 0; i < N; i++) {
                //cout << "[" << i << "]=" << h_output[i] << endl;
            }
        }

        // bench mark using half8
        {
            cl::Buffer d_samplefp16 = cl::Buffer(context, h_samplefp16.begin(), h_samplefp16.end(), true);
            cl::Buffer d_outputfp16 = cl::Buffer(context,
                                             CL_MEM_READ_WRITE,
                                             sizeof(h_samplefp16[0]) * h_samplefp16.size()
                                             );

            cl::make_kernel<int, cl::Buffer, cl::Buffer> acorr_hf8( program, "acorr_hf8");

            double start_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0;
            for(int i = 0; i < LOOPS; i++) {
                // compute
                acorr_hf8(cl::EnqueueArgs( queue, global, local),
                      N,
                      d_samplefp16,
                      d_outputfp16
                      );
                queue.finish();
            }
            double run_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0 - start_time;

            std::vector<uint16_t> h_outputfp16( N, 0);
            cl::copy(queue, d_outputfp16, h_outputfp16.begin(), h_outputfp16.end());

            cout << "half8 result ----" << endl;
            cout << run_time << " sec" << endl;
            for(int i = 0; i < N; i++) {
                //cout << "[" << i << "]=" << h_output[i] << endl;
            }
        }

        // bench mark using half16
        {
            cl::Buffer d_samplefp16 = cl::Buffer(context, h_samplefp16.begin(), h_samplefp16.end(), true);
            cl::Buffer d_outputfp16 = cl::Buffer(context,
                                             CL_MEM_READ_WRITE,
                                             sizeof(h_samplefp16[0]) * h_samplefp16.size()
                                             );

            cl::make_kernel<int, cl::Buffer, cl::Buffer> acorr_hf16( program, "acorr_hf16");

            double start_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0;
            for(int i = 0; i < LOOPS; i++) {
                // compute
                acorr_hf16(cl::EnqueueArgs( queue, global, local),
                      N,
                      d_samplefp16,
                      d_outputfp16
                      );
                queue.finish();
            }
            double run_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0 - start_time;

            std::vector<uint16_t> h_outputfp16( N, 0);
            cl::copy(queue, d_outputfp16, h_outputfp16.begin(), h_outputfp16.end());

            cout << "half16 result ----" << endl;
            cout << run_time << " sec" << endl;
            for(int i = 0; i < N; i++) {
                //cout << "[" << i << "]=" << h_output[i] << endl;
            }
        }

        // bench mark using half16*2
        {
            cl::Buffer d_samplefp16 = cl::Buffer(context, h_samplefp16.begin(), h_samplefp16.end(), true);
            cl::Buffer d_outputfp16 = cl::Buffer(context,
                                             CL_MEM_READ_WRITE,
                                             sizeof(h_samplefp16[0]) * h_samplefp16.size()
                                             );

            cl::make_kernel<int, cl::Buffer, cl::Buffer> acorr_hf32( program, "acorr_hf16");

            double start_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0;
            for(int i = 0; i < LOOPS; i++) {
                // compute
                acorr_hf32(cl::EnqueueArgs( queue, global, local),
                      N,
                      d_outputfp16,
                      d_outputfp16
                      );
                queue.finish();
            }
            double run_time   = static_cast<double>( timer.getTimeMilliseconds()) / 1000.0 - start_time;

            std::vector<uint16_t> h_outputfp16( N, 0);
            cl::copy(queue, d_outputfp16, h_outputfp16.begin(), h_outputfp16.end());

            cout << "half16*2 result ----" << endl;
            cout << run_time << " sec" << endl;
            for(int i = 0; i < N; i++) {
                //cout << "[" << i << "]=" << h_output[i] << endl;
            }
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
