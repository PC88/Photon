#pragma once
#include "RT1W/AABB.h"
#include "RT1W/ray.h"

class material;

struct hit_record 
{
	point3 p;
	vec3 normal;
	std::shared_ptr<material> mat_ptr;
	double t;

	double u;
	double v;

	bool front_face;

	inline void set_face_normal(const ray& r, const vec3& outward_normal)
	{
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable
{
public:
	__device__ __host__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
	__device__ __host__ virtual bool bounding_box(double t0, double t1, AABB& output_box) const = 0;
};

