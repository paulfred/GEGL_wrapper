
#define __CL_ENABLE_EXCEPTIONS 1
#include <CL/cl.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "util_opencl.h"

using namespace cl;
using std::string;
using std::vector;
using std::cout;
using std::cerr;

const bool debug = false;

const char* src = "__kernel void hello(__global float4* out, \
					__global const float4* in) \
{ \
	int id = get_global_id(0) * 4; \
 \
	out[id].xyz = 1.0 - in[id].xyz; \
	out[id].w = in[id].w; \
}";

Program::Sources getSource(const string& fname) {
	/*std::ifstream infile(fname.c_str());
	if (!infile) {
		throw std::exception();
	}
	std::stringstream buffer;
	buffer << infile.rdbuf();
	infile.close();

	string src = buffer.str();
	cerr << "Loaded program (" << src.length() << ")\n";
	return Program::Sources(1, std::make_pair(src.c_str(), src.length()));
	*/
	return Program::Sources(1, std::make_pair(src, strlen(src)));
}

void RunProgram(GEGLclass* ggObj, float* in, float* out) {

	cl_int err = CL_SUCCESS;

	try {

		vector<Platform> platforms;
		Platform::get(&platforms);
		if (platforms.size() == 0) {
			cerr << "No platforms\n";
			exit(-1);
		}
		Platform p = platforms[0];
		if(debug) {		
			cout << "Found " << platforms.size() << " platforms\n";
			cout << p.getInfo<CL_PLATFORM_PROFILE>() << "\n"
				<< p.getInfo<CL_PLATFORM_NAME>() << "\n"
				<< p.getInfo<CL_PLATFORM_VENDOR>() << "\n"
				<< p.getInfo<CL_PLATFORM_VERSION>() << "\n"
				<< p.getInfo<CL_PLATFORM_EXTENSIONS>() << "\n";
		}

		cl_context_properties properties[] = 
		  { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

		Context context(CL_DEVICE_TYPE_CPU, properties);

		vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		cout << "Found " << devices.size() << " devices\n";

		Program::Sources source = getSource("GEGL_invertgamma.cl");
		Program prg(context, source);
		try {
			prg.build(devices);
		} catch(cl::Error er) {
			if (er.err() == CL_BUILD_PROGRAM_FAILURE) {
				::size_t log_size = 0;
				clGetProgramBuildInfo((cl_program&)prg, (cl_device_id&)devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
				char* log = (char*)new char[log_size];
				clGetProgramBuildInfo((cl_program&)prg, (cl_device_id&)devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
				// cerr << prg.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log_size);
				cerr << log;
				delete[] log;
			}
		}

		Kernel kernel(prg, "hello", &err);

		CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
		Event event;

		::size_t bufsize = get_pixelcount(ggObj) * sizeof(float);
		Buffer inbuf(context, CL_MEM_READ_ONLY, bufsize);
		Buffer outbuf(context, CL_MEM_WRITE_ONLY, bufsize);

		queue.enqueueWriteBuffer(inbuf, CL_FALSE, 0, bufsize, in);

		kernel.setArg(0, outbuf);
		kernel.setArg(1, inbuf);

		queue.enqueueNDRangeKernel(
		  kernel, 
		  NullRange, 
		  NDRange(bufsize),
		  NullRange,
		  NULL,
		  &event); 

		event.wait();

		queue.enqueueReadBuffer(outbuf, CL_TRUE, 0, bufsize, out);
	}
	#ifdef __CL_ENABLE_EXCEPTIONS
	catch (cl::Error err) {
		cerr << "ERROR: "<< err.what()
		  << "(" << err.err() << ")" << std::endl;
	}
	#endif
	catch(...) {
		cerr << "Unexpected error!";
	}
}