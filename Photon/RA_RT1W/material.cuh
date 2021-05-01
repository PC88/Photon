#pragma once

#include "RA_RT1W\ray.cuh"
#include "RA_RT1W\hitable.cuh"
#include <curand_kernel.h>

namespace RA
{
	__device__ float schlick(float cosine, float ref_idx);

	__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted);

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

	__device__ vec3 random_in_unit_sphere(curandState* local_rand_state);

	__device__ vec3 reflect(const vec3& v, const vec3& n);
}

namespace RA
{
	class material
	{
	public:
		__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
	};

	class lambertian : public material
	{
	public:
		__device__ lambertian(const vec3& a);
		__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const;

		vec3 albedo;
	};

	class metal : public material
	{
	public:
		__device__ metal(const vec3& a, float f);
		__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const;
		vec3 albedo;
		float fuzz;
	};

	class dielectric : public material
	{
	public:
		__device__ dielectric(float ri);
		__device__ virtual bool scatter(const ray& r_in,
			const hit_record& rec,
			vec3& attenuation,
			ray& scattered,
			curandState* local_rand_state) const;

		float ref_idx;
	};
}