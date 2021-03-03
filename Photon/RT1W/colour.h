#pragma once
#include "RT1W/vec3.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

// global for now - might be changed later its simply for speed
// static as linkage issues would arise
static void write_color(std::ostream &out, color pixel_color, int samples_per_pixel)
{
	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();

	// Divide the color by the number of samples and gamma-correct for gamma=2.0.
	auto scale = 1.0 / samples_per_pixel;

	// refactored to use std::sqrt
	r = std::sqrt(scale * r);
	g = std::sqrt(scale * g);
	b = std::sqrt(scale * b);

	// Write the translated [0,255] value of each color component.
	out << static_cast<int>(256 * UtilityManager::instance().clamp(r, 0.0, 0.999)) << ' '
		<< static_cast<int>(256 * UtilityManager::instance().clamp(g, 0.0, 0.999)) << ' '
		<< static_cast<int>(256 * UtilityManager::instance().clamp(b, 0.0, 0.999)) << '\n';
}