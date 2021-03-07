#pragma once
#include "RT1W\vec3.h"
#include "RT1W\textures\solid_color.h"

class checker_texture :
	public base_texture
{
public:
	checker_texture() {}

	checker_texture(std::shared_ptr<base_texture> t0, std::shared_ptr<base_texture> t1)
		: even(t0), odd(t1) {}

	checker_texture(color c1, color c2)
		: even(std::make_shared<solid_color>(c1)), odd(std::make_shared<solid_color>(c2)) {}

	virtual color value(double u, double v, const point3& p) const override;

public:
	std::shared_ptr<base_texture> odd;
	std::shared_ptr<base_texture> even;

};
