#pragma once
#include "RA_RT1W\hitable.cuh"

// credit Roger Allen for the original
// credit: https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

namespace RA
{
	class hitable_list : public hitable
	{
	public:
		__device__ hitable_list();
		__device__ hitable_list(hitable** l, int n);
		__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
		hitable** list;
		int list_size;
	};

}