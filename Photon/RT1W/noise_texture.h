#pragma once
#include "texture.h"

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
