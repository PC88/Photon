#pragma once
#include <cuda_runtime.h>

class ray
{
public:
	__host__ __device__ ray() {}
	__host__ __device__ ray(const point3& origin, const vec3& direction, double time = 0.0)
		: orig(origin), dir(direction), tm(time)
	{}

	__host__ __device__ point3 origin() const { return orig; }
	__host__ __device__ vec3 direction() const { return dir; }
	__host__ __device__ double time() const { return tm; }

	__host__ __device__ point3 at(double t) const
	{
		return orig + t * dir;
	}

public:
	point3 orig;
	vec3 dir;
	double tm;
};

