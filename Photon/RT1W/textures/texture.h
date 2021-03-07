#pragma once
#include "RT1W/vec3.h"


// base texture renamed, utterly infuriating
// https://stackoverflow.com/questions/64345442/cuda-c-class-is-mistaken-for-template

class base_texture 
{
public:
	virtual color value(double u, double v, const point3& p) const = 0;
};