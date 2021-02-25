#include "RT1W/lambertian.h"

bool lambertian::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const
{
	vec3 scatter_direction = rec.normal + UtilityManager::instance().random_unit_vector();
	scattered = ray(rec.p, scatter_direction, r_in.time());
	attenuation = albedo->value(rec.u, rec.v, rec.p);
	return true;
}