#pragma once

#include "RT1W\hittables\hittable.h"

#include <memory>
#include <vector>


class hittable_list : public hittable 
{
public:
	hittable_list() {}
	hittable_list(std::shared_ptr<hittable> object) { add(object); }

	void clear() { objects.clear(); }
	void add(std::shared_ptr<hittable> object) { objects.push_back(object); }

	virtual bool hit(
		const ray& r, double tmin, double tmax, hit_record& rec) const override;

	virtual bool bounding_box(double t0, double t1, AABB& output_box) const override;

public:
	std::vector<std::shared_ptr<hittable>> objects;
};