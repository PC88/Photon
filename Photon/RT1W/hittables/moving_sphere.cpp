#include "RT1W\hittables\moving_sphere.h"

point3 moving_sphere::center(double time) const
{
	return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

bool moving_sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
	vec3 oc = r.origin() - center(r.time());
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;

	auto discriminant = half_b * half_b - a * c;

	if (discriminant > 0)
	{
		auto root = sqrt(discriminant);

		auto temp = (-half_b - root) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.p = r.at(rec.t);
			auto outward_normal = (rec.p - center(r.time())) / radius;
			rec.set_face_normal(r, outward_normal);
			rec.mat_ptr = mat_ptr;
			return true;
		}

		temp = (-half_b + root) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.p = r.at(rec.t);
			auto outward_normal = (rec.p - center(r.time())) / radius;
			rec.set_face_normal(r, outward_normal);
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

bool moving_sphere::bounding_box(double t0, double t1, AABB& output_box) const
{
	AABB box0(
		center(t0) - vec3(radius, radius, radius),
		center(t0) + vec3(radius, radius, radius));
	AABB box1(
		center(t1) - vec3(radius, radius, radius),
		center(t1) + vec3(radius, radius, radius));
	output_box = UtilityManager::instance().surrounding_box(box0, box1);
	return true;
}