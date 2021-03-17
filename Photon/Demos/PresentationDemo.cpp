#include "PresentationDemo.h"
#include "RT1W/colour.h"

PresentationDemo::PresentationDemo()
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

	world = final_scene();
	aspect_ratio = 1.0;
	image_width = 800;
	samples_per_pixel = 1000;
	background = color(0, 0, 0);
	lookfrom = point3(478, 278, -600);
	lookat = point3(278, 278, 0);
	vfov = 40.0;

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
			write_color_ppm(pixel_color, samples_per_pixel, outputData, img);
		}
	}

	img.write("test.ppm", outputData);

	std::cerr << "\nDone.\n";

}

PresentationDemo::~PresentationDemo()
{

}

void PresentationDemo::Update(double interval)
{

}

void PresentationDemo::ImGuiRender()
{

}

void PresentationDemo::Render()
{

}

double PresentationDemo::hit_sphere(const point3& center, double radius, const ray& r)
{
	vec3 oc = r.origin() - center;

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
		return (-half_b - std::sqrt(discriminant)) / a;
	}
}

color PresentationDemo::ray_color(const ray& r, const color& background, const hittable& world, int depth)
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

hittable_list PresentationDemo::final_scene()
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
			auto y1 = 1.0;//UtilityManager::instance().random_double(1, 101);
			auto z1 = z0 + w;

			boxes1.add(std::make_shared<box>(point3(x0, y0, z0), point3(x1, y1, z1), ground));
		}
	}

	hittable_list objects;

	objects.add(std::make_shared<bvh_node>(boxes1, 0, 1));

	auto light = std::make_shared<diffuse_light>(color(7, 7, 7));
	objects.add(std::make_shared<xz_rect>(123, 423, 147, 412, 554, light));

	objects.add(std::make_shared<sphere>(point3(160, 50, 400), 50, std::make_shared<dielectric>(1.5)));

	auto boundary = std::make_shared<sphere>(point3(0, 0, 0), 5000, std::make_shared<dielectric>(1.5));
	objects.add(std::make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));

	// the 'H' in hello
	{
		objects.add(std::make_shared<xy_rect>(400, 450, 0, 200, 554, light));
		objects.add(std::make_shared<xy_rect>(300, 450, 75, 125, 554, light));
		objects.add(std::make_shared<xy_rect>(300, 350, 0, 200, 554, light));
	}
	// the 'E' in hello
	{
		objects.add(std::make_shared<xy_rect>(200, 250, 0, 200, 554, light));

		objects.add(std::make_shared<xy_rect>(140, 200, 150, 200, 554, light));
		objects.add(std::make_shared<xy_rect>(140, 200, 75, 125, 554, light));
		objects.add(std::make_shared<xy_rect>(140, 200, 0, 50, 554, light));
	}
	// the 'L'&'L' in hello
	{
		objects.add(std::make_shared<xy_rect>(0, 90, 0, 50, 554, light));
		objects.add(std::make_shared<xy_rect>(40, 90, 0, 200, 554, light));

		objects.add(std::make_shared<xy_rect>(-140, -50, 0, 50, 554, light));
		objects.add(std::make_shared<xy_rect>(-90, -50, 0, 200, 554, light));
	}
	// the 'O' in hello
	{
		objects.add(std::make_shared<xy_rect>(-240, -190, 0, 200, 554, light));
		objects.add(std::make_shared<xy_rect>(-340, -190, 150, 200, 554, light));

		objects.add(std::make_shared<xy_rect>(-340, -190, 0, 50, 554, light));
		objects.add(std::make_shared<xy_rect>(-340, -290, 0, 200, 554, light));
	}

	return objects;

}
