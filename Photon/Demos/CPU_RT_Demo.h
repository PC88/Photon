#pragma once
#include "Demo.h"

#include "RT1W/vec3.h"
#include "RT1W/ray.h"
#include "RT1W/hittable.h"
#include "RT1W/sphere.h"
// added
#include "RT1W/translate.h"
#include "RT1W/rotate_y.h"
// added
#include "RT1W/hittable_list.h"
#include "RT1W/camera.h"


#include "RT1W/image_texture.h"
#include "RT1W/noise_texture.h"
#include "RT1W/solid_color.h"
#include "RT1W/texture.h"

#include "RT1W/metal.h"


class CPU_RT_Demo :
    public Demo
{
};

