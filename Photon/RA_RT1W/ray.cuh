#pragma once
#include "RA_RT1W\vec3.cuh"

namespace RA
{


	class ray
	{
	public:
		__device__ ray();
		__device__ ray(const vec3& a, const vec3& b);
		__device__ vec3 origin() const;
		__device__ vec3 direction() const;
		__device__ vec3 point_at_parameter(float t) const;

		vec3 A;
		vec3 B;
	};


}
