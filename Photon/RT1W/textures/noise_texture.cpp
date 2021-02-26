#include "RT1W\textures\noise_texture.h"

color noise_texture::value(double u, double v, const point3& p) const
{
	return color(1, 1, 1) * 0.5 * (1 + sin(scale * p.z() + 10 * noise.turb(p)));
}
