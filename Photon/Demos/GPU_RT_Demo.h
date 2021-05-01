#include "Demo.h"
#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>

// Roger Allen classes, ported to more modular type and within namespace

#include "RA_RT1W\vec3.cuh"
#include "RA_RT1W\ray.cuh"
#include "RA_RT1W\sphere.cuh"
#include "RA_RT1W\hitable_list.cuh"
#include "RA_RT1W\camera.cuh"
#include "RA_RT1W\material.cuh"


class GPU_RT_Demo :
    public Demo
{
public:
    GPU_RT_Demo();
    virtual ~GPU_RT_Demo();
};