#pragma once
#include "RT1W\material.h"

class dielectric :
	public material
{

public:
	dielectric(double ri) : ref_idx(ri) {}

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override;

	double ref_idx;
};
