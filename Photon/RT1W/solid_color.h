#pragma once
#include "texture.h"

class solid_color : public texture
{
public:
	solid_color() {}
	solid_color(color c) : color_value(c) {}

	solid_color(double red, double green, double blue)
		: solid_color(color(red, green, blue)) {}

	virtual color value(double u, double v, const vec3& p) const override;

private:
	color color_value;
};

