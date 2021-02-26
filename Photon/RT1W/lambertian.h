#pragma once
#include "RT1W/material.h"
#include "RT1W\textures\texture.h"

struct hit_record;

class lambertian : public material
{
public:
	lambertian(const color& a) : albedo(std::make_shared<solid_color>(a)) {}
	lambertian(std::shared_ptr<texture> a) : albedo(a) {}

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override;

public:
	std::shared_ptr<texture> albedo;
};

