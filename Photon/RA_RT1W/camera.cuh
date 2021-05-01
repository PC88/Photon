#pragma once
#include <curand_kernel.h>
#include "RA_RT1W\ray.cuh"

namespace RA
{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

	__device__ vec3 random_in_unit_disk(curandState* local_rand_state);

	class camera
	{
	public:
		__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist);
		__device__ ray get_ray(float s, float t, curandState* local_rand_state);

		vec3 origin;
		vec3 lower_left_corner;
		vec3 horizontal;
		vec3 vertical;
		vec3 u, v, w;
		float lens_radius;
	};

}