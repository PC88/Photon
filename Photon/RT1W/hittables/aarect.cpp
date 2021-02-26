#include "RT1W\hittables\aarect.h"


bool xy_rect::bounding_box(double t0, double t1, AABB& output_box) const
{
	// The bounding box must have non-zero width in each dimension, so pad the Z
	// dimension a small amount.
	output_box = AABB(point3(x0, y0, k - 0.0001), point3(x1, y1, k + 0.0001));
	return true;
}

bool xz_rect::bounding_box(double t0, double t1, AABB& output_box) const
{
	// The bounding box must have non-zero width in each dimension, so pad the Y
	// dimension a small amount.
	output_box = AABB(point3(x0, k - 0.0001, z0), point3(x1, k + 0.0001, z1));
	return true;
}

bool yz_rect::bounding_box(double t0, double t1, AABB& output_box) const
{
	// The bounding box must have non-zero width in each dimension, so pad the X
	// dimension a small amount.
	output_box = AABB(point3(k - 0.0001, y0, z0), point3(k + 0.0001, y1, z1));
	return true;
}



bool xy_rect::hit(const ray& r, double t0, double t1, hit_record& rec) const
{
	auto t = (k - r.origin().z()) / r.direction().z();
	if (t < t0 || t > t1)
		return false;
	auto x = r.origin().x() + t * r.direction().x();
	auto y = r.origin().y() + t * r.direction().y();
	if (x < x0 || x > x1 || y < y0 || y > y1)
		return false;
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (y - y0) / (y1 - y0);
	rec.t = t;
	auto outward_normal = vec3(0, 0, 1);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);
	return true;
}

bool xz_rect::hit(const ray& r, double t0, double t1, hit_record& rec) const
{
	auto t = (k - r.origin().y()) / r.direction().y();
	if (t < t0 || t > t1)
		return false;
	auto x = r.origin().x() + t * r.direction().x();
	auto z = r.origin().z() + t * r.direction().z();
	if (x < x0 || x > x1 || z < z0 || z > z1)
		return false;
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	auto outward_normal = vec3(0, 1, 0);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);
	return true;
}

bool yz_rect::hit(const ray& r, double t0, double t1, hit_record& rec) const
{
	auto t = (k - r.origin().x()) / r.direction().x();
	if (t < t0 || t > t1)
		return false;
	auto y = r.origin().y() + t * r.direction().y();
	auto z = r.origin().z() + t * r.direction().z();
	if (y < y0 || y > y1 || z < z0 || z > z1)
		return false;
	rec.u = (y - y0) / (y1 - y0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	auto outward_normal = vec3(1, 0, 0);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);
	return true;
}