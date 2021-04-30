#pragma once

#include "RT1W\hittables\hittable.h"
#include "RT1W\materials\material.h"
#include "RT1W\textures\texture.h"

class constant_medium : public hittable 
{
public:
	__device__ __host__ constant_medium(std::shared_ptr<hittable> b, double d, std::shared_ptr<base_texture> a)
		: boundary(b),
		neg_inv_density(-1 / d),
		phase_function(std::make_shared<isotropic>(a))
	{}

	__device__ __host__ constant_medium(std::shared_ptr<hittable> b, double d, color c)
		: boundary(b),
		neg_inv_density(-1 / d),
		phase_function(std::make_shared<isotropic>(c))
	{}

	__device__ __host__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;

	__device__ __host__ virtual bool bounding_box(double t0, double t1, AABB& output_box) const override;

public:
	std::shared_ptr<hittable> boundary;
	std::shared_ptr<material> phase_function;
	double neg_inv_density;
};