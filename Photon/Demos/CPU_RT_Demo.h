#pragma once
#include "Demo.h"

#include "RT1W\vec3.h"
#include "RT1W\ray.h"
#include "RT1W\hittables\hittable.h"
#include "RT1W\hittables\sphere.h"
// added
#include "RT1W\hittables\translate.h"
#include "RT1W\hittables\rotate_y.h"
// added
#include "RT1W\hittables\hittable_list.h"
#include "RT1W/camera.h"


#include "RT1W\textures\image_texture.h"
#include "RT1W\textures\noise_texture.h"
#include "RT1W\textures\solid_color.h"
#include "RT1W\textures\texture.h"

#include "RT1W\materials\metal.h"
#include "RT1W\materials\lambertian.h"
#include "RT1W\materials\dielectric.h"
#include "RT1W\hittables\moving_sphere.h"

#include "RT1W\textures\checker_texture.h"
#include "RT1W\hittables\aarect.h"
#include "RT1W\hittables\box.h"
#include "RT1W\hittables\constant_medium.h"
#include "RT1W\hittables\bvh.h"


class CPU_RT_Demo :
    public Demo
{
	CPU_RT_Demo();
	virtual ~CPU_RT_Demo();

	// Optional inherited functions
	virtual void Update(double interval) override;
	virtual void ImGuiRender() override;
	virtual void Render() override;

    double hit_sphere(const point3& center, double radius, const ray& r);

    // depth is added here to stop the recursions from blowing the stack
    color ray_color(const ray& r, const color& background, const hittable& world, int depth);

    // cover image function
    hittable_list random_scene();
    hittable_list two_spheres();
    hittable_list two_perlin_spheres();
    hittable_list earth();
    hittable_list simple_light();
    hittable_list cornell_box();
    hittable_list cornell_smoke();
    hittable_list final_scene();
};

