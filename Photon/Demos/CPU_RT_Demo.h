#pragma once
#include "Demo.h"

#include "RT1W/vec3.h"
#include "RT1W/ray.h"
#include "RT1W\hittables\hittable.h"
#include "RT1W\hittables\sphere.h"
// added
#include "RT1W\hittables\translate.h"
#include "RT1W\hittables\rotate_y.h"
// added
#include "RT1W\hittables\hittable_list.h"
#include "RT1W/camera.h"


#include "RT1W/image_texture.h"
#include "RT1W/noise_texture.h"
#include "RT1W/solid_color.h"
#include "RT1W/texture.h"

#include "RT1W/metal.h"
#include "RT1W/lambertian.h"
#include "RT1W/dielectric.h"
#include "RT1W\hittables\moving_sphere.h"

#include "RT1W/checker_texture.h"
#include "RT1W\hittables\aarect.h"
#include "RT1W\hittables\box.h"
#include "RT1W\hittables\constant_medium.h"
#include "RT1W\hittables\bvh.h"


class CPU_RT_Demo :
    public Demo
{
};

