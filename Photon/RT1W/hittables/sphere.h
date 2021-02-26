#pragma once


#include "RT1W\hittables\hittable.h"
#include "RT1W\vec3.h"

class hittable;

class sphere : public hittable 
{
public:
	sphere() {}
	sphere(point3 cen, double r, std::shared_ptr<material> m) : center(cen), radius(r), mat_ptr(m) {};

	virtual bool hit(
		const ray& r, double tmin, double tmax, hit_record& rec) const override;

	virtual bool bounding_box(double t0, double t1, AABB& output_box) const override;
public:
	point3 center;
	double radius;
	std::shared_ptr<material> mat_ptr;
};

