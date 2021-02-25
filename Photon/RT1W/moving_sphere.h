#pragma once
#include "hittable.h"

class moving_sphere :
	public hittable
{
public:
	moving_sphere() {}
	moving_sphere(
		point3 cen0, point3 cen1, double t0, double t1, double r, std::shared_ptr<material> m)
		: center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m)
	{};

	virtual bool hit(
		const ray& r, double tmin, double tmax, hit_record& rec) const override;

	virtual bool bounding_box(double t0, double t1, AABB& output_box) const override;

	point3 center(double time) const;

public:
	point3 center0, center1;
	double time0, time1;
	double radius;
	std::shared_ptr<material> mat_ptr;
};

