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


class GPU_RT_Demo :
    public Demo
{
public:
    GPU_RT_Demo();
    virtual ~GPU_RT_Demo();
};