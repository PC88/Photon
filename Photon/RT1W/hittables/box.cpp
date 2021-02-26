#include "RT1W\hittables\box.h"



box::box(const point3& p0, const point3& p1, std::shared_ptr<material> ptr)
{
	box_min = p0;
	box_max = p1;

	sides.add(std::make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
	sides.add(std::make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

	sides.add(std::make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
	sides.add(std::make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

	sides.add(std::make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
	sides.add(std::make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
}

bool box::hit(const ray& r, double t0, double t1, hit_record& rec) const
{
	return sides.hit(r, t0, t1, rec);
}

bool box::bounding_box(double t0, double t1, AABB& output_box) const
{
	output_box = AABB(box_min, box_max);
	return true;
}
