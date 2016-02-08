
__kernel void myfilter(	__global float4* out,
						__global const float4* in,
						__global int* ids)
{
	int id = get_global_id(0);
	if (id > ids[0]) ids[0] = id;

	out[id].xyz = 1.0 - in[id].xyz;
	out[id].w = in[id].w;
}
