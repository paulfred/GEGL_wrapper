
__kernel void myfilter(	__global float4* out,
						__global const float4* in,
						__global float* params,
						__global int* ids)
{
	__local float4 m[3];
	float scale = 1.0 - params[0];

	m[0].x = 0.393 + 0.607 * scale;
	m[0].y = 0.769 - 0.769 * scale;
	m[0].z = 0.189 - 0.189 * scale;

	m[1].x = 0.349 - 0.349 * scale;
	m[1].y = 0.686 + 0.314 * scale;
	m[1].z = 0.168 - 0.168 * scale;

	m[2].x = 0.272 - 0.272 * scale;
	m[2].y = 0.534 - 0.534 * scale;
	m[2].z = 0.131 + 0.869 * scale;

	int id = get_global_id(0);
	if (id > ids[0]) ids[0] = id;

	float4 r = m[0] * in[id];
	out[id].x = r.x + r.y + r.z;
	r = m[1] * in[id];
	out[id].y = r.x + r.y + r.z;
	r = m[2] * in[id];
	out[id].z = r.x + r.y + r.z;
	out[id].w = in[id].w;
}
