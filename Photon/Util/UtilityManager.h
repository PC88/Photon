#pragma once

#include <iostream>
#include <string>

#include <cuda_runtime.h>
// includes moved here from rtweekend.h
#include <memory>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
// includes moved here from rtweekend.h

class vec3; // forward dec
class AABB;
class hittable;

class UtilityManager
{
public:
	static UtilityManager& instance();

	UtilityManager(UtilityManager const&) = delete;
	void operator=(UtilityManager const&) = delete;

	// Constants
	const double infinity = std::numeric_limits<double>::infinity();
	const double pi = 3.1415926535897932385;

	// Utility Functions
	static inline double degrees_to_radians(double degrees)
	{
		return degrees * UtilityManager::instance().pi / 180.0;
	}

	static __device__ __host__ inline double random_double()
	{
		// Returns a random real in [0,1).
		return rand() / (RAND_MAX + 1.0);
	}

	static __device__ __host__ inline double random_double(double min, double max)
	{
		// Returns a random real in [min,max).
		return min + (max - min) * random_double();
	}

	static inline double clamp(double x, double min, double max)
	{
		if (x < min) return min;
		if (x > max) return max;
		return x;
	}

	static __device__ __host__ inline int random_int(int min, int max)
	{
		// Returns a random integer in [min,max].
		return static_cast<int>(random_double(min, max + 1));
	}

	///////////////////////////// MOVED FROM VEC 3.H //
	vec3 random_in_unit_disk();

	vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat);

	// "The reflected ray direction in red is just v+2b. In our design"
	vec3 reflect(const vec3& v, const vec3& n);

	vec3 random_in_hemisphere(const vec3& normal);

	vec3 random_unit_vector();

	vec3 random_in_unit_sphere();
	///////////////////////////// MOVED FROM SPHERE.H//

	void get_sphere_uv(const vec3& p, double& u, double& v);

	///////////////////////////// MOVED FROM AABB.H//
	AABB surrounding_box(AABB box0, AABB box1);

	///////////////////////////// MOVED FROM MATERIAL.H//
	double schlick(double cosine, double ref_idx);


	inline bool box_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b, int axis);

	bool box_x_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b);

	bool box_y_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b);

	bool box_z_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b);

private:
	UtilityManager();
	virtual ~UtilityManager();
};


///////////////////////////// MOVED FROM BVH.H// - global still for now, should be reworked later.
inline bool box_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b, int axis);

bool box_x_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b);

bool box_y_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b);

bool box_z_compare(const std::shared_ptr<hittable> a, const std::shared_ptr<hittable> b);

//quick test for first version of parallel
static inline double random_double()
{
	// Returns a random real in [0,1).
	return rand() / (RAND_MAX + 1.0);
}

void write_color_ppm(vec3 pixel_color, int samples_per_pixel, std::vector<unsigned char>& data);

void write_color(std::ostream& out, vec3 pixel_color, int samples_per_pixel);