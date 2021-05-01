﻿#include "Demos\GPU_RT_Demo.h"
#include <curand_kernel.h> // cuRAND
#include "ppm/ppm.hpp"

// test commit GPU

// commit to mark reverting to alternate form,

// credit: https://github.com/rogerallen/raytracinginoneweekendincuda/tree/master
// credit: https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) 
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = 2.0f * dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4.0f * a * c;
	return (discriminant > 0.0f);
}

__device__ vec3 colour(const ray& r) 
{
	if (hit_sphere(vec3(0, 0, -1), 0.5, r))
	{
		return vec3(1, 0, 0);
	}
	vec3 unit_direction = unit_vector(r.direction());

	float t = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3* fb, int max_x, int max_y,
	vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) 
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, lower_left_corner + u * horizontal + v * vertical);
	fb[pixel_index] = colour(r);
}

GPU_RT_Demo::GPU_RT_Demo()
{
	int nx = 1200;
	int ny = 600;
	int tx = 8;
	int ty = 8;

	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	// allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render<<<blocks, threads>>>(fb, nx, ny,
		vec3(-2.0, -1.0, -1.0),
		vec3(4.0, 0.0, 0.0),
		vec3(0.0, 2.0, 0.0),
		vec3(0.0, 0.0, 0.0));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";


	ppm img;
	img.w = nx;
	img.h = ny;
	img.magic = "P3";
	img.max = 255;
	img.capacity = img.w * img.h * img.nchannels;
	std::vector<unsigned char> outputData;


	std::vector<unsigned char> data;
	// Output FB as Image
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) 
	{
		std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
		for (int i = 0; i < nx; i++) 
		{
			size_t pixel_index = j * nx + i;
			int ir = int(255.99 * fb[pixel_index].r());
			int ig = int(255.99 * fb[pixel_index].g());
			int ib = int(255.99 * fb[pixel_index].b());
			//std::cout << ir << " " << ig << " " << ib << "\n";


			//data.resize(img.w * img.h * img.nchannels);
			data.push_back(static_cast<int>(ir));
			data.push_back(static_cast<int>(ig));
			data.push_back(static_cast<int>(ib));
		}
	}

	img.write("gpu_test.ppm", data);

	checkCudaErrors(cudaFree(fb));
}

GPU_RT_Demo::~GPU_RT_Demo()
{

}

