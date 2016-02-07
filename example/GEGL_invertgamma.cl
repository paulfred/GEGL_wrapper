
__kernel void hello(__global float4* out,
					__global const float4* in)
{
	int id = get_global_id(0);

	out[id].xyz = 1.0 - in[id].xyz;
	out[id].w = in[id].w;
}
