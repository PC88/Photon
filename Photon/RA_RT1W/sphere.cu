﻿#include "RA_RT1W\sphere.cuh"

// credit Roger Allen for the original
// credit: https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

namespace RA
{

	__device__ sphere::sphere()
	{

	}

	__device__ sphere::sphere(vec3 cen, float r, material* m)
		: center(cen), radius(r), mat_ptr(m)
	{

	}

	__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
	{
		vec3 oc = r.origin() - center;
		float a = dot(r.direction(), r.direction());
		float b = dot(oc, r.direction());
		float c = dot(oc, oc) - radius * radius;
		float discriminant = b * b - a * c;
		if (discriminant > 0) {
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.mat_ptr = mat_ptr;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.mat_ptr = mat_ptr;
				return true;
			}
		}
		return false;
	}
}