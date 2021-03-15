#include "RT1W\hittables\bvh.h"

bool bvh_node::bounding_box(double t0, double t1, AABB& output_box) const
{
	output_box = box;
	return true;
}

bool bvh_node::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
	if (!box.hit(r, t_min, t_max))
		return false;

	bool hit_left = left->hit(r, t_min, t_max, rec);
	bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

	return hit_left || hit_right;
}

bvh_node::bvh_node(std::vector<std::shared_ptr<hittable>>& objects,
	size_t start, size_t end, double time0, double time1)
{
	const int axis = UtilityManager::instance().random_int(0, 2);

	// comparator, this is a function of. 
	// std compare https://en.cppreference.com/w/cpp/named_req/Compare
	// function pointer type being utilized to confer the expression at the end.

	auto comparator = (axis == 0) ? box_x_compare
		: (axis == 1) ? box_y_compare
		: box_z_compare;

	//auto comparator = [&](const auto a, const auto b) // shared ptrs hittable
	//{return(axis == 0) ? UtilityManager::instance().box_x_compare(a,b)
	//		: (axis == 1) ? UtilityManager::instance().box_y_compare(a,b)
	//		: UtilityManager::instance().box_z_compare(a,b); };

	//auto comparator = [&](const auto a, const auto b) // more concise version of above - note actually an object
	//{
	//	return UtilityManager::instance().box_compare(a, b, axis);
	//};


	size_t object_span = end - start;

	if (object_span == 1)
	{
		left = right = objects[start];
	}
	else if (object_span == 2)
	{
		if (comparator(objects[start], objects[start + 1]))
		{
			left = objects[start];
			right = objects[start + 1];
		}
		else
		{
			left = objects[start + 1];
			right = objects[start];
		}
	}
	else
	{
		std::sort(objects.begin() + start, objects.begin() + end, comparator);

		auto mid = start + object_span / 2;
		left = std::make_shared<bvh_node>(objects, start, mid, time0, time1);
		right = std::make_shared<bvh_node>(objects, mid, end, time0, time1);
	}

	AABB box_left, box_right;

	if (!left->bounding_box(time0, time1, box_left)
		|| !right->bounding_box(time0, time1, box_right)
		)
	{
		std::cerr << "No bounding box in bvh_node constructor.\n";
	}

	box = UtilityManager::instance().surrounding_box(box_left, box_right);
}