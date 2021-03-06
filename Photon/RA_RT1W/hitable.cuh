﻿#pragma once
#include "RA_RT1W\ray.cuh"

// credit Roger Allen for the original
// credit: https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

namespace RA
{
	class material;
}


namespace RA
{
	struct hit_record
	{
		float t;
		vec3 p;
		vec3 normal;
		material* mat_ptr;
	};

	class hitable
	{
	public:
		__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
	};

}