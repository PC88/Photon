#include "RT1W\materials\dielectric.h"

bool dielectric::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const
{
	attenuation = color(1.0, 1.0, 1.0);
	double etai_over_etat = rec.front_face ? (1.0 / ref_idx) : ref_idx;

	vec3 unit_direction = unit_vector(r_in.direction());

	double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
	double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
	if (etai_over_etat * sin_theta > 1.0)
	{
		vec3 reflected = UtilityManager::instance().reflect(unit_direction, rec.normal);
		scattered = ray(rec.p, reflected);
		return true;
	}
	double reflect_prob = UtilityManager::instance().schlick(cos_theta, etai_over_etat);
	if (UtilityManager::instance().random_double() < reflect_prob)
	{
		vec3 reflected = UtilityManager::instance().reflect(unit_direction, rec.normal);
		scattered = ray(rec.p, reflected);
		return true;
	}
	vec3 refracted = UtilityManager::instance().refract(unit_direction, rec.normal, etai_over_etat);
	scattered = ray(rec.p, refracted);
	return true;
}
