#pragma once
#include "RT1W\textures\texture.h"
#include "RT1W/ray.h"
#include "RT1W\hittables\hittable.h"
#include "RT1W\textures\solid_color.h"

class material
{
public:
	// not pure virtual: non enforced interface
	virtual color emitted(double u, double v, const point3& p) const 
	{
		return color(0, 0, 0);
	}

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};


class diffuse_light : public material 
{
public:
	diffuse_light(std::shared_ptr<texture> a) : emit(a) {}
	diffuse_light(color c) : emit(std::make_shared<solid_color>(c)) {}

	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
	) const override 
	{
		return false;
	}

	virtual color emitted(double u, double v, const point3& p) const override 
	{
		return emit->value(u, v, p);
	}

public:
	std::shared_ptr<texture> emit;
};

class isotropic : public material 
{
public:
	isotropic(color c) : albedo(std::make_shared<solid_color>(c)) {}
	isotropic(std::shared_ptr<texture> a) : albedo(a) {}

	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
	) const override 
	{
		scattered = ray(rec.p, UtilityManager::instance().random_in_unit_sphere(), r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

public:
	std::shared_ptr<texture> albedo;
};