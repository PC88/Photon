#pragma once
#include "Demos\Demo.h"
#include "Util\UtilityManager.h"
#include "RT1W\vec3.h"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand_kernel.h> // cuRAND

class RA_GPU_RT_Demo :
    public Demo
{
	RA_GPU_RT_Demo();
	virtual ~RA_GPU_RT_Demo();
};

