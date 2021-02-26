#include "RT1W/metal.h"
#include "RT1W\hittables\hittable.h"

bool metal::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const
{
	vec3 reflected = UtilityManager::instance().reflect(unit_vector(r_in.direction()), rec.normal);
	scattered = ray(rec.p, reflected + fuzz * UtilityManager::instance().random_in_unit_sphere());
	attenuation = albedo;
	return (dot(scattered.direction(), rec.normal) > 0);
}
