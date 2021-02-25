#pragma once
#include "RT1W\hittable.h"

class rotate_y : public hittable 
{
public:
	rotate_y(std::shared_ptr<hittable> p, double angle);
	virtual ~rotate_y()
	{

	}

	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;

	virtual bool bounding_box(double t0, double t1, AABB& output_box) const override;

public:
	std::shared_ptr<hittable> ptr;
	double sin_theta;
	double cos_theta;
	bool hasbox;
	AABB bbox;
};