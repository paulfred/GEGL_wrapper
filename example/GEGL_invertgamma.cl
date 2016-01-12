
__kernel void hello(__global float* out,
					__global const float* in)
{
	int id = get_global_id(0) * 4;

	out[id].xyz = 1.0 - in[id].xyz;
	out[id].a = in[id].a;
}
