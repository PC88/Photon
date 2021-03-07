#pragma once
#include <cmath>
#include <iostream>
#include "Util/UtilityManager.h"

#include <cuda_runtime.h>

using std::sqrt;

// include commented out above due to errors, the linkage of globals is global, meaning:
// the definition of the function can be linked, but the declaration cannot - hence the second definition here - PC
// this is now being extended to these stubs - which are moved to rtweekend.h
//double random_double();
//double random_double(double min, double max);


class vec3
{
public:

public:
	__host__ __device__ vec3() : e{ 0,0,0 } {}
	__host__ __device__ vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}

	// these were in-lined
	__host__ __device__ double x() const { return e[0]; }
	__host__ __device__ double y() const { return e[1]; }
	__host__ __device__ double z() const { return e[2]; }
	__host__ __device__ float r() const { return e[0]; }
	__host__ __device__ float g() const { return e[1]; }
	__host__ __device__ float b() const { return e[2]; }

	__host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ double operator[](int i) const { return e[i]; }
	__host__ __device__ double& operator[](int i) { return e[i]; }

	__host__ __device__ vec3& operator+=(const vec3 &v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	__host__ __device__ vec3& operator*=(const double t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	__host__ __device__ vec3& operator/=(const double t)
	{
		return *this *= 1 / t;
	}

	__host__ __device__ double length() const
	{
		return sqrt(length_squared());
	}

	__host__ __device__ double length_squared() const
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	inline static vec3 random()
	{
		return vec3(UtilityManager::instance().random_double(),
			UtilityManager::instance().random_double(),
			UtilityManager::instance().random_double());
	}

	inline static vec3 random(double min, double max)
	{
		return vec3(UtilityManager::instance().random_double(min, max),
			UtilityManager::instance().random_double(min, max), 
			UtilityManager::instance().random_double(min, max));
	}


public:
	double e[3];

};

// vec3 Utility Functions

inline std::ostream& operator<<(std::ostream &out, const vec3 &v)
{
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3 &v)
{
	return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, double t)
{
	return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, double t)
{
	return (1 / t) * v;
}

__host__ __device__ inline double dot(const vec3 &u, const vec3 &v)
{
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
	return v / v.length();
}


// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB

// needs to be defined below the .h - re-factored
//vec3 random_in_unit_disk();
//vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat);
//vec3 reflect(const vec3& v, const vec3& n);
//vec3 random_in_hemisphere(const vec3& normal);
//vec3 random_unit_vector();
//vec3 random_in_unit_sphere();