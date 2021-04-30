#pragma once
#include "RT1W\hittables\hittable.h"

class rotate_y : public hittable 
{
public:
	__device__ __host__ rotate_y(std::shared_ptr<hittable> p, double angle);
	__device__ __host__ virtual ~rotate_y()
	{

	}

	__device__ __host__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;

	__device__ __host__ virtual bool bounding_box(double t0, double t1, AABB& output_box) const override;

public:
	std::shared_ptr<hittable> ptr;
	double sin_theta;
	double cos_theta;
	bool hasbox;
	AABB bbox;
};