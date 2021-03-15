#include "UtilityManager.h"
#include "RT1W/vec3.h"
#include "RT1W/AABB.h"
#include "RT1W\hittables\hittable.h"

// base includes from RT1W utils
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

UtilityManager& UtilityManager::instance()
{
	static UtilityManager _self;
	return _self;
}

UtilityManager::UtilityManager()
{

}



vec3 UtilityManager::random_in_unit_disk()
{
	while (true)
	{
		auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

vec3 UtilityManager::refract(const vec3& uv, const vec3& n, double etai_over_etat)
{
	auto cos_theta = dot(-uv, n);
	vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}

// "The reflected ray direction in red is just v+2b. In our design"
vec3 UtilityManager::reflect(const vec3& v, const vec3& n)
{
	return v - 2 * dot(v, n) * n;
}

vec3 UtilityManager::random_in_hemisphere(const vec3& normal)
{
	vec3 in_unit_sphere = random_in_unit_sphere();
	if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

vec3 UtilityManager::random_unit_vector()
{
	// this is not optimal definition, but it will do for now
	const double pi = 3.1415926535897932385;
	auto a = random_double(0, 2 * pi);
	auto z = random_double(-1, 1);
	auto r = sqrt(1 - z * z);
	return vec3(r * cos(a), r * sin(a), z);
}

vec3 UtilityManager::random_in_unit_sphere()
{
	while (true)
	{
		auto p = vec3::random(-1, 1);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

void UtilityManager::get_sphere_uv(const vec3& p, double& u, double& v)
{
	auto phi = atan2(p.z(), p.x());
	auto theta = asin(p.y());
	u = 1 - (phi + pi) / (2 * pi);
	v = (theta + pi / 2) / pi;
}

AABB UtilityManager::surrounding_box(AABB box0, AABB box1)
{
	point3 small(fmin(box0.min().x(), box1.min().x()),
		fmin(box0.min().y(), box1.min().y()),
		fmin(box0.min().z(), box1.min().z()));

	point3 big(fmax(box0.max().x(), box1.max().x()),
		fmax(box0.max().y(), box1.max().y()),
		fmax(box0.max().z(), box1.max().z()));

	return AABB(small, big);
}


double UtilityManager::schlick(double cosine, double ref_idx)
{
	auto r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}

bool UtilityManager::box_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b, int axis)
{
	AABB box_a;
	AABB box_b;

	if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
	{
		std::cerr << "No bounding box in bvh_node constructor.\n";
	}

	return box_a.min().e[axis] < box_b.min().e[axis];
}

bool UtilityManager::box_x_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b)
{
	return box_compare(a, b, 0);
}

bool UtilityManager::box_y_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b)
{
	return box_compare(a, b, 1);
}

bool UtilityManager::box_z_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b)
{
	return box_compare(a, b, 2);
}

UtilityManager::~UtilityManager()
{

}

bool box_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b, int axis)
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
