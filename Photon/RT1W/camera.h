#pragma once
#include "RT1W/vec3.h"
#include "RT1W/ray.h"

class camera
{
public:
	camera(point3 lookfrom, point3 lookat, vec3 vup,
		double vfov,// vertical field-of-view in degrees
		double aspect_ratio,
		double aperture,
		double focus_dist,
		double t0 = 0,
		double t1 = 0);

	ray get_ray(double s, double t) const;

private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	double lens_radius;
	double time0, time1;  // shutter open/close times
};