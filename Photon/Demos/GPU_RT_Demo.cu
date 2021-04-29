#include "Demos\GPU_RT_Demo.h"
#include <curand_kernel.h> // cuRAND
#include "ppm/ppm.hpp"

// test commit GPU

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

__device__ vec3 colour(const ray& r) 
{
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3* fb, int max_x, int max_y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	fb[pixel_index] = vec3(float(i) / max_x, float(j) / max_y, 0.2f);
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
	render<<<blocks, threads>>>(fb, nx, ny);
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
		for (int i = 0; i < nx; i++) 
		{
			size_t pixel_index = j * nx + i;
			int ir = int(255.99 * fb[pixel_index].r());
			int ig = int(255.99 * fb[pixel_index].g());
			int ib = int(255.99 * fb[pixel_index].b());
			//std::cout << ir << " " << ig << " " << ib << "\n";

			// Write the translated [0,255] value of each color component.

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

