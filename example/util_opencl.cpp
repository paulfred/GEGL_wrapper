
#define __CL_ENABLE_EXCEPTIONS 1
#include <CL/cl.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>

#include "util_opencl.h"

using namespace cl;
using std::string;
using std::vector;
using std::cout;
using std::cerr;

const bool debug = false;

string safe_echo(char s) {
	switch(s) {
		case '\t': return "\\t";
		case '\r': return "\\r";
		case '\n': return "\\n";
		default: break;
	}
	return string(1,s);
}

double time_in_ms(Event& ev) {
	return (ev.getProfilingInfo<CL_PROFILING_COMMAND_END>()
	 - ev.getProfilingInfo<CL_PROFILING_COMMAND_START>())/1000000.0;
}

class ProgramCache {
	std::vector<string> _fnames;
	std::vector<string> _sources;

public:
	const string& update(const string& fname, const string& src) {
		assert(_fnames.size() == _sources.size());
		std::vector<string>::iterator it;
		it = std::find(_fnames.begin(), _fnames.end(), fname);
		if (it == _fnames.end()) {
			_fnames.push_back(fname);
			_sources.push_back(src);
			return src;
		}
		return *(_sources.begin() + (it - _fnames.begin()));
	}

	Program::Sources load(const string& fname) {
		std::ifstream infile(fname.c_str());
		if (!infile) {
			throw Error(1, "Source file not found");
		}

		std::stringstream buffer;
		buffer << infile.rdbuf();
		infile.close();

		string src = buffer.str();
		
		cerr << "Loaded program (" << src.length() << ")\n";

		// this returns a string that goes out of scope
		//return Program::Sources(1, std::make_pair(src.c_str(), src.length()));
		const string& csrc = update(fname, src);
		return Program::Sources(1, std::make_pair(csrc.c_str(), csrc.length()));
	}
};

ProgramCache cache;

string getSource(const string& fname) {
	std::ifstream infile(fname.c_str());
	if (!infile) {
		throw Error(1, "Source file not found");
	}

	std::stringstream buffer;
	buffer << infile.rdbuf();
	infile.close();

	return buffer.str();
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

		string src = getSource("GEGL_invertgamma.cl");
		Program::Sources source = Program::Sources(1, std::make_pair(src.c_str(), src.length()));
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

		Kernel kernel(prg, "myfilter", &err);

		CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
		Event event;
		Event write_in;
		Event read_out;

		::size_t bufsize = get_pixelcount(ggObj) * sizeof(float) * 4;
		Buffer inbuf(context, CL_MEM_READ_ONLY, bufsize);
		Buffer outbuf(context, CL_MEM_WRITE_ONLY, bufsize);
		Buffer scratch(context, CL_MEM_READ_WRITE, 32);

		int* scratch_buf = (int*)malloc(32);
		memset(scratch_buf, 0, 8);

		queue.enqueueWriteBuffer(inbuf, CL_TRUE, 0, bufsize, in, NULL, &write_in);
		queue.enqueueWriteBuffer(scratch, CL_TRUE, 0, 32, scratch_buf);

		kernel.setArg(0, outbuf);
		kernel.setArg(1, inbuf);
		kernel.setArg(2, scratch);

		queue.enqueueNDRangeKernel(
		  kernel, 
		  NullRange, 
		  NDRange(get_pixelcount(ggObj)),
		  NullRange,
		  NULL,
		  &event);

		event.wait();
		queue.enqueueReadBuffer(scratch, CL_TRUE, 0, 8, scratch_buf);
		queue.enqueueReadBuffer(outbuf, CL_TRUE, 0, bufsize, out, NULL, &read_out);

		cerr << "max work item " << scratch_buf[0] << "\n";
		cerr << std::setprecision(3) << std::fixed
			<< "Kernel execution took " << time_in_ms(event) << "ms\n"
			<< "Writing input took " << time_in_ms(write_in) << "ms\n"
			<< "Reading output took " << time_in_ms(read_out) << "ms\n";
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