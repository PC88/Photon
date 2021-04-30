#pragma once

#include "RT1W\hittables\hittable.h"

#include <memory>
#include <vector>


class hittable_list : public hittable 
{
public:
	__device__ __host__ hittable_list() {}
	__device__ __host__ hittable_list(std::shared_ptr<hittable> object) { add(object); }

	__device__ __host__ void clear() { objects.clear(); }
	__device__ __host__ void add(std::shared_ptr<hittable> object) { objects.push_back(object); }

	__device__ __host__ virtual bool hit(
		const ray& r, double tmin, double tmax, hit_record& rec) const override;

	__device__ __host__ virtual bool bounding_box(double t0, double t1, AABB& output_box) const override;

public:
	std::vector<std::shared_ptr<hittable>> objects;
};