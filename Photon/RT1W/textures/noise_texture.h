#pragma once
#include "RT1W\textures\texture.h"
#include "RT1W\perlin.h"

class noise_texture : public texture
{
public:
	noise_texture() {}
	noise_texture(double sc) : scale(sc) {}

	virtual color value(double u, double v, const point3& p) const override;

public:
	perlin noise;
	double scale;
};
