#pragma once
#include <algorithm>
#include "RT1W/hittable_list.h"

class bvh_node : public hittable 
{
public:
	bvh_node();

	bvh_node(hittable_list& list, double time0, double time1)
		: bvh_node(list.objects, 0, list.objects.size(), time0, time1)
	{}

	bvh_node(
		std::vector<std::shared_ptr<hittable>>& objects,
		size_t start, size_t end, double time0, double time1);

	virtual bool hit(
		const ray& r, double tmin, double tmax, hit_record& rec) const override;

	virtual bool bounding_box(double t0, double t1, AABB& output_box) const override;

public:
	std::shared_ptr<hittable> left;
	std::shared_ptr<hittable> right;
	AABB box;
};

inline bool box_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b, int axis) 
{
	AABB box_a;
	AABB box_b;

	if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
	{
		std::cerr << "No bounding box in bvh_node constructor.\n";
	}

	return box_a.min().e[axis] < box_b.min().e[axis];
}


bool box_x_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b) 
{
	return box_compare(a, b, 0);
}

bool box_y_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b) 
{
	return box_compare(a, b, 1);
}

bool box_z_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b) 
{
	return box_compare(a, b, 2);
}