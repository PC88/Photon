#include "RA_RT1W\ray.cuh"
namespace RA
{

	__device__ ray::ray()
	{

	}

	__device__ ray::ray(const vec3& a, const vec3& b)
	{
		A = a;
		B = b;
	}


	__device__ vec3 ray::origin() const
	{
		return A;
	}

	__device__ vec3 ray::direction() const
	{
		return B;
	}

	__device__ vec3 ray::point_at_parameter(float t) const
	{
		return A + t * B;
	}

}