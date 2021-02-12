#pragma once
#include "Demo.h"

#include "iostream"
#include "RT1W/vec3.h"
#include "RT1W/ray.h"
#include "RT1W/hittable.h"
#include "RT1W/sphere.h"
#include "RT1W/hittable_list.h"
#include "RT1W/camera.h"
#include "RT1W/colour.h"
#include "RT1W/metal.h"
#include "RT1W/lambertian.h"
#include "RT1W/dielectric.h"
#include "RT1W/moving_sphere.h"
#include "RT1W/checker_texture.h"
#include "RT1W/aarect.h"
#include "RT1W/box.h"
#include "RT1W/constant_medium.h"
#include "RT1W/bvh.h"

class CPU_RT_Demo :
    public Demo
{
};

