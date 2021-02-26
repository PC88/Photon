#pragma once
#include "RT1W\materials\material.h"
#include "RT1W\ray.h"
#include "RT1W\vec3.h"

struct hit_record;

class metal :
	public material
{
public:
	metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}


	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override;

public:
	color albedo;
	double fuzz;
};