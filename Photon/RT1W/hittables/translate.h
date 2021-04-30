#pragma once
#include "Util\UtilityManager.h"
#include "RT1W\hittables\hittable.h"

class translate : public hittable
{
public:
	__device__ __host__ translate(std::shared_ptr<hittable> p, const vec3& displacement)
		: ptr(p), offset(displacement) {}

	__device__ __host__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;

	__device__ __host__ virtual bool bounding_box(double t0, double t1, AABB& output_box) const override;

	__device__ __host__ virtual ~translate()
	{

	}
public:
	std::shared_ptr<hittable> ptr;
	vec3 offset;
};