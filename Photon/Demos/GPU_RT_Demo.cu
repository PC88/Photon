#include "GPU_RT_Demo.h"
#include "RT1W/colour.h" // output function
#include <curand_kernel.h> // cuRAND

// test commit GPU

// credit: https://github.com/rogerallen/raytracinginoneweekendincuda/tree/master
// credit: https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) 
{
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

GPU_RT_Demo::GPU_RT_Demo()
{
	// Image
	auto aspect_ratio = 16.0 / 9.0;
	int image_width = 400;
	int samples_per_pixel = 10; // was 100
	const int max_depth = 50;

	// World
	hittable_list world;

	point3 lookfrom;
	point3 lookat;
	auto vfov = 40.0;
	auto aperture = 0.0;
	color background(0, 0, 0);

	switch (8)
	{
	case 1:
		world = random_scene();
		background = color(0.70, 0.80, 1.00);
		lookfrom = point3(13, 2, 3);
		lookat = point3(0, 0, 0);
		vfov = 20.0;
		aperture = 0.1;
		break;

	default:
	case 2:
		world = two_spheres();
		background = color(0.70, 0.80, 1.00);
		lookfrom = point3(13, 2, 3);
		lookat = point3(0, 0, 0);
		vfov = 20.0;
		break;
	case 3:
		world = two_perlin_spheres();
		background = color(0.70, 0.80, 1.00);
		lookfrom = point3(13, 2, 3);
		lookat = point3(0, 0, 0);
		vfov = 20.0;
		break;
	case 4:
		world = earth();
		background = color(0.70, 0.80, 1.00);
		lookfrom = point3(13, 2, 3);
		lookat = point3(0, 0, 0);
		vfov = 20.0;
		break;
	case 5:
		world = simple_light();
		samples_per_pixel = 400;
		background = color(0, 0, 0);
		lookfrom = point3(26, 3, 6);
		lookat = point3(0, 2, 0);
		vfov = 20.0;
		break;
	case 6:
		world = cornell_box();
		aspect_ratio = 1.0;
		image_width = 600;
		samples_per_pixel = 200;
		background = color(0, 0, 0);
		lookfrom = point3(278, 278, -800);
		lookat = point3(278, 278, 0);
		vfov = 40.0;
		break;
	case 7:
		world = cornell_smoke();
		aspect_ratio = 1.0;
		image_width = 600;
		samples_per_pixel = 200;
		lookfrom = point3(278, 278, -800);
		lookat = point3(278, 278, 0);
		vfov = 40.0;
		break;
	case 8:
		world = final_scene();
		aspect_ratio = 1.0;
		image_width = 800;
		samples_per_pixel = 10;
		background = color(0, 0, 0);
		lookfrom = point3(478, 278, -600);
		lookat = point3(278, 278, 0);
		vfov = 40.0;
		break;
	}

	// Camera
	vec3 vup(0, 1, 0);
	auto dist_to_focus = 10.0;
	int image_height = static_cast<int>(image_width / aspect_ratio);

	camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

	ppm img;
	img.w = image_width;
	img.h = image_height;
	img.magic = "P3";
	img.max = 255;
	img.capacity = img.w * img.h * img.nchannels;
	std::vector<unsigned char> outputData;
	// Render
	std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

	for (int j = image_height - 1; j >= 0; --j)
	{
		std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
		for (int i = 0; i < image_width; ++i)
		{
			color pixel_color(0, 0, 0);
			for (int s = 0; s < samples_per_pixel; ++s)
			{
				auto u = (i + UtilityManager::instance().random_double()) / (image_width - 1);
				auto v = (j + UtilityManager::instance().random_double()) / (image_height - 1);
				ray r = cam.get_ray(u, v);
				pixel_color += ray_color(r, background, world, max_depth);
			}
			//write_color(std::cout, pixel_color, samples_per_pixel);
			write_color_ppm(pixel_color, samples_per_pixel, outputData, img);
		}
	}

	img.write("test.ppm", outputData);

	std::cerr << "\nDone.\n";
}


void GPU_RT_Demo::initCuda(int width, int height)
{
	int num_pixels = width * height;
	size_t fb_size = 3 * num_pixels * sizeof(float);

	// allocate FB
	float* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
}

void GPU_RT_Demo::cleanCuda()
{

}


GPU_RT_Demo::~GPU_RT_Demo()
{

}

void GPU_RT_Demo::Update(double interval)
{

}

void GPU_RT_Demo::ImGuiRender()
{

}

void GPU_RT_Demo::Render()
{

}

double GPU_RT_Demo::hit_sphere(const point3& center, double radius, const ray& r)
{
	vec3 oc = r.origin() - center;
	/// refactor 
	//auto a = dot(r.direction(), r.direction());
	//auto b = 2.0 * dot(oc, r.direction());
	//auto c = dot(oc, oc) - radius * radius;
	//auto discriminant = b * b - 4 * a*c;

	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant < 0)
	{
		return -1.0;
	}
	else
	{
		//return (-b - sqrt(discriminant)) / (2.0*a);
		return (-half_b - std::sqrt(discriminant)) / a;
	}
}

color GPU_RT_Demo::ray_color(const ray& r, const color& background, const hittable& world, int depth)
{
	hit_record rec;

	// If we've exceeded the ray bounce limit, no more light is gathered.
	if (depth <= 0)
	{
		return color(0, 0, 0);
	}

	// If the ray hits nothing, return the background color.
	if (!world.hit(r, 0.001, UtilityManager::instance().infinity, rec))
	{
		return background;
	}

	ray scattered;
	color attenuation;
	color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

	if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
	{
		return emitted;
	}

	return emitted + attenuation * ray_color(scattered, background, world, depth - 1);

	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

hittable_list GPU_RT_Demo::random_scene()
{
	hittable_list world;

	auto checker = std::make_shared<checker_texture>(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
	world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, std::make_shared<lambertian>(checker)));

	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			auto choose_mat = UtilityManager::instance().random_double();
			point3 center(a + 0.9 * UtilityManager::instance().random_double(),
				0.2, b + 0.9 * UtilityManager::instance().random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9)
			{
				std::shared_ptr<material> sphere_material;

				if (choose_mat < 0.8)
				{
					// diffuse
					auto albedo = color::random() * color::random();
					sphere_material = std::make_shared<lambertian>(albedo);
					world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
					auto center2 = center + vec3(0, UtilityManager::instance().random_double(0, .5), 0);
					world.add(std::make_shared<moving_sphere>(center, center2, 0.0, 1.0, 0.2, sphere_material));
				}
				else if (choose_mat < 0.95)
				{
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = UtilityManager::instance().random_double(0, 0.5);
					sphere_material = std::make_shared<metal>(albedo, fuzz);
					world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
				}
				else
				{
					// glass
					sphere_material = std::make_shared<dielectric>(1.5);
					world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
				}
			}
		}
	}

	auto material1 = std::make_shared<dielectric>(1.5);
	world.add(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

	auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

	auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

	return world;
}

hittable_list GPU_RT_Demo::two_spheres()
{
	hittable_list objects;

	auto checker = std::make_shared<checker_texture>(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));

	objects.add(std::make_shared<sphere>(point3(0, -10, 0), 10, std::make_shared<lambertian>(checker)));
	objects.add(std::make_shared<sphere>(point3(0, 10, 0), 10, std::make_shared<lambertian>(checker)));

	return objects;
}

hittable_list GPU_RT_Demo::two_perlin_spheres()
{
	hittable_list objects;
	auto pertext = std::make_shared<noise_texture>(4);
	objects.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, std::make_shared<lambertian>(pertext)));
	objects.add(std::make_shared<sphere>(point3(0, 2, 0), 2, std::make_shared<lambertian>(pertext)));

	return objects;
}

hittable_list GPU_RT_Demo::earth()
{
	auto earth_texture = std::make_shared<image_texture>("earthmap.jpg");
	auto earth_surface = std::make_shared<lambertian>(earth_texture);
	auto globe = std::make_shared<sphere>(point3(0, 0, 0), 2, earth_surface);

	return hittable_list(globe);
}

hittable_list GPU_RT_Demo::simple_light()
{
	hittable_list objects;

	auto pertext = std::make_shared<noise_texture>(4);
	objects.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, std::make_shared<lambertian>(pertext)));
	objects.add(std::make_shared<sphere>(point3(0, 2, 0), 2, std::make_shared<lambertian>(pertext)));

	auto difflight = std::make_shared<diffuse_light>(color(4, 4, 4));
	objects.add(std::make_shared<xy_rect>(3, 5, 1, 3, -2, difflight));

	auto diffspherelight = std::make_shared<diffuse_light>(color(4, 4, 4));
	objects.add(std::make_shared<sphere>(point3(0, 7, 0), 2, diffspherelight));

	return objects;
}

hittable_list GPU_RT_Demo::cornell_box()
{
	hittable_list objects;

	auto red = std::make_shared<lambertian>(color(.65, .05, .05));
	auto white = std::make_shared<lambertian>(color(.73, .73, .73));
	auto green = std::make_shared<lambertian>(color(.12, .45, .15));
	auto light = std::make_shared<diffuse_light>(color(15, 15, 15));

	objects.add(std::make_shared<yz_rect>(0, 555, 0, 555, 555, green));
	objects.add(std::make_shared<yz_rect>(0, 555, 0, 555, 0, red));
	objects.add(std::make_shared<xz_rect>(213, 343, 227, 332, 554, light));
	objects.add(std::make_shared<xz_rect>(0, 555, 0, 555, 0, white));
	objects.add(std::make_shared<xz_rect>(0, 555, 0, 555, 555, white));
	objects.add(std::make_shared<xy_rect>(0, 555, 0, 555, 555, white));

	// boxes
	std::shared_ptr<hittable> box1 = std::make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), white);
	box1 = std::make_shared<rotate_y>(box1, 15);
	box1 = std::make_shared<translate>(box1, vec3(265, 0, 295));
	objects.add(box1);

	std::shared_ptr<hittable> box2 = std::make_shared<box>(point3(0, 0, 0), point3(165, 165, 165), white);
	box2 = std::make_shared<rotate_y>(box2, -18);
	box2 = std::make_shared<translate>(box2, vec3(130, 0, 65));
	objects.add(box2);

	return objects;
}

hittable_list GPU_RT_Demo::cornell_smoke()
{
	hittable_list objects;

	auto red = std::make_shared<lambertian>(color(.65, .05, .05));
	auto white = std::make_shared<lambertian>(color(.73, .73, .73));
	auto green = std::make_shared<lambertian>(color(.12, .45, .15));
	auto light = std::make_shared<diffuse_light>(color(7, 7, 7));

	objects.add(std::make_shared<yz_rect>(0, 555, 0, 555, 555, green));
	objects.add(std::make_shared<yz_rect>(0, 555, 0, 555, 0, red));
	objects.add(std::make_shared<xz_rect>(113, 443, 127, 432, 554, light));
	objects.add(std::make_shared<xz_rect>(0, 555, 0, 555, 555, white));
	objects.add(std::make_shared<xz_rect>(0, 555, 0, 555, 0, white));
	objects.add(std::make_shared<xy_rect>(0, 555, 0, 555, 555, white));

	std::shared_ptr<hittable> box1 = std::make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), white);
	box1 = std::make_shared<rotate_y>(box1, 15);
	box1 = std::make_shared<translate>(box1, vec3(265, 0, 295));

	std::shared_ptr<hittable> box2 = std::make_shared<box>(point3(0, 0, 0), point3(165, 165, 165), white);
	box2 = std::make_shared<rotate_y>(box2, -18);
	box2 = std::make_shared<translate>(box2, vec3(130, 0, 65));

	objects.add(std::make_shared<constant_medium>(box1, 0.01, color(0, 0, 0)));
	objects.add(std::make_shared<constant_medium>(box2, 0.01, color(1, 1, 1)));

	return objects;
}

hittable_list GPU_RT_Demo::final_scene()
{
	hittable_list boxes1;
	auto ground = std::make_shared<lambertian>(color(0.48, 0.83, 0.53));

	const int boxes_per_side = 20;
	for (int i = 0; i < boxes_per_side; i++)
	{
		for (int j = 0; j < boxes_per_side; j++)
		{
			auto w = 100.0;
			auto x0 = -1000.0 + i * w;
			auto z0 = -1000.0 + j * w;
			auto y0 = 0.0;
			auto x1 = x0 + w;
			auto y1 = UtilityManager::instance().random_double(1, 101);
			auto z1 = z0 + w;

			boxes1.add(std::make_shared<box>(point3(x0, y0, z0), point3(x1, y1, z1), ground));
		}
	}

	hittable_list objects;

	objects.add(std::make_shared<bvh_node>(boxes1, 0, 1));

	auto light = std::make_shared<diffuse_light>(color(7, 7, 7));
	objects.add(std::make_shared<xz_rect>(123, 423, 147, 412, 554, light));

	auto center1 = point3(400, 400, 200);
	auto center2 = center1 + vec3(30, 0, 0);
	auto moving_sphere_material = std::make_shared<lambertian>(color(0.7, 0.3, 0.1));
	objects.add(std::make_shared<moving_sphere>(center1, center2, 0, 1, 50, moving_sphere_material));

	objects.add(std::make_shared<sphere>(point3(260, 150, 45), 50, std::make_shared<dielectric>(1.5)));
	objects.add(std::make_shared<sphere>(
		point3(0, 150, 145), 50, std::make_shared<metal>(color(0.8, 0.8, 0.9), 10.0)
		));

	auto boundary = std::make_shared<sphere>(point3(360, 150, 145), 70, std::make_shared<dielectric>(1.5));
	objects.add(boundary);
	objects.add(std::make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
	boundary = std::make_shared<sphere>(point3(0, 0, 0), 5000, std::make_shared<dielectric>(1.5));
	objects.add(std::make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));

	auto emat = std::make_shared<lambertian>(std::make_shared<image_texture>("earthmap.jpg"));
	objects.add(std::make_shared<sphere>(point3(400, 200, 400), 100, emat));
	auto pertext = std::make_shared<noise_texture>(0.1);
	objects.add(std::make_shared<sphere>(point3(220, 280, 300), 80, std::make_shared<lambertian>(pertext)));

	hittable_list boxes2;
	auto white = std::make_shared<lambertian>(color(.73, .73, .73));
	int ns = 1000;
	for (int j = 0; j < ns; j++) {
		boxes2.add(std::make_shared<sphere>(point3::random(0, 165), 10, white));
	}

	objects.add(std::make_shared<translate>(
		std::make_shared<rotate_y>(
			std::make_shared<bvh_node>(boxes2, 0.0, 1.0), 15),
		vec3(-100, 270, 395)
		)
	);

	return objects;
}

