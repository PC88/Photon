#pragma once

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>


// Usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) 
{
	return degrees * pi / 180.0;
}

inline double random_double() 
{
	// Returns a random real in [0,1).
	return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) 
{
	// Returns a random real in [min,max).
	return min + (max - min)*random_double();
}

inline double clamp(double x, double min, double max) 
{
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

inline int random_int(int min, int max) 
{
	// Returns a random integer in [min,max].
	return static_cast<int>(random_double(min, max + 1));
}


///////////////////////////////////////////////////
///////////////////////////// MOVED FROM VEC 3.H //
///////////////////////////////////////////////////

vec3 random_in_unit_disk()
{
	while (true)
	{
		auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat)
{
	auto cos_theta = dot(-uv, n);
	vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}

// "The reflected ray direction in red is just v+2b. In our design"
vec3 reflect(const vec3& v, const vec3& n)
{
	return v - 2 * dot(v, n) * n;
}

vec3 random_in_hemisphere(const vec3& normal)
{
	vec3 in_unit_sphere = random_in_unit_sphere();
	if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

vec3 random_unit_vector()
{
	// this is not optimal definition, but it will do for now
	const double pi = 3.1415926535897932385;

	auto a = random_double(0, 2 * pi);
	auto z = random_double(-1, 1);
	auto r = sqrt(1 - z * z);
	return vec3(r * cos(a), r * sin(a), z);
}


vec3 random_in_unit_sphere()
{
	while (true)
	{
		auto p = vec3::random(-1, 1);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

///////////////////////////////////////////////////
///////////////////////////// MOVED FROM VEC 3.H //
///////////////////////////////////////////////////