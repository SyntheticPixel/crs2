// Defs
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes
#include "main.h"

using namespace std;
using namespace crs;
using namespace rapidjson;

// main
int main(int argc, const char * argv[]){
	Document		dom;
	string			jsonfile;
	string			output;
	
	CudaContext		cc;
	
	Camera 			*host_camera;
	Camera			*device_camera;
	
	Sphere			*host_spheres;
	Sphere			*device_spheres;
	
	BxdfTable		bxdfTable;
	Bxdf			*host_bxdfs;
	Bxdf			*device_bxdfs;

	float 			gamma_correction;

	unsigned int spherecount = 0;
	unsigned int bxdfcount = 0;

	if(argc == 2){
		jsonfile = argv[1];
	}else{
		cout << " ./crs -path-to-json-file" << std::endl;
		return EXIT_FAILURE;
	}

	// Print out some stats
	cc.GetDeviceProps();
	if(cc.state == CRS_NO_ERROR){
		cc.PrintCudaContext();
	}else{
		cc.EvaluateState();
		return EXIT_SUCCESS;
	}

	// Read the json file
	cout << " Opening scene description: " << jsonfile << std::endl;
	ifstream ifs(jsonfile);
	IStreamWrapper isw(ifs);
	dom.ParseStream(isw);

	if(dom.HasParseError()){
		ParseErrorCode e = dom.GetParseError();
		cout << " Error parsing json file, error #" << e << std::endl;
		return EXIT_FAILURE;
	}

	if(dom.HasMember("rendersettings")){
		cout << " Rendersettings found..." << std::endl;
		// read the rendersettings
		Value *setting;
		setting = Pointer("/rendersettings/output").Get(dom);
		output = setting->GetString();
		
		setting = Pointer("/rendersettings/width").Get(dom);
		cc.width = setting->GetInt();
		cout << " Image width : " << cc.width << std::endl;
		
		setting = Pointer("/rendersettings/height").Get(dom);
		cc.height = setting->GetInt();
		cout << " Image height : " << cc.height << std::endl;
		
		setting = Pointer("/rendersettings/samples").Get(dom);
		cc.samples = setting->GetInt();
		cout << " Render samples : " << cc.samples << std::endl;
		
		setting = Pointer("/rendersettings/depth").Get(dom);
		cc.depth = setting->GetInt();
		cout << " Path depth : " << cc.depth << std::endl;
		
		setting = Pointer("/rendersettings/gamma").Get(dom);
		gamma_correction = setting->GetFloat();
		cout << " Gamma correction : " << gamma_correction << std::endl;
	}else{
		cout << " Error parsing scene file, no rendersettings found!" << std::endl;
		return EXIT_FAILURE;
	}

	if(dom.HasMember("camera")){
		cout << " Camera found..." << std::endl;

		// read the camera settings
		host_camera = new Camera;

		host_camera->resolution_x = (float)cc.width;
		host_camera->resolution_y = (float)cc.height;
		
		Value *setting;

		setting = Pointer("/camera/position/0").Get(dom);
		host_camera->position.x = setting->GetFloat();
		setting = Pointer("/camera/position/1").Get(dom);
		host_camera->position.y = setting->GetFloat();
		setting = Pointer("/camera/position/2").Get(dom);
		host_camera->position.z = setting->GetFloat();

		setting = Pointer("/camera/lookat/0").Get(dom);
		host_camera->lookat.x = setting->GetFloat();
		setting = Pointer("/camera/lookat/1").Get(dom);
		host_camera->lookat.y = setting->GetFloat();
		setting = Pointer("/camera/lookat/2").Get(dom);
		host_camera->lookat.z = setting->GetFloat();

		setting = Pointer("/camera/up/0").Get(dom);
		host_camera->up.x = setting->GetFloat();
		setting = Pointer("/camera/up/1").Get(dom);
		host_camera->up.y = setting->GetFloat();
		setting = Pointer("/camera/up/2").Get(dom);
		host_camera->up.z = setting->GetFloat();

		setting = Pointer("/camera/field_of_view").Get(dom);
		host_camera->fov = setting->GetFloat();

		setting = Pointer("/camera/focus_distance").Get(dom);
		host_camera->focus_distance = setting->GetFloat();

		setting = Pointer("/camera/aperture_radius").Get(dom);
		host_camera->aperture_radius = setting->GetFloat();

		// make the camera current
		host_camera->update();

	}else{
		cout << " Error parsing scene file, no camera found!" << std::endl;
		return EXIT_FAILURE;
	}

	if(dom.HasMember("bxdfs")){
		Value *setting;
		setting = Pointer("/bxdfs").Get(dom);
		assert(setting->IsArray());
		cout << " " << setting->Size() << " Bxdf(s) found..." << std::endl;

		bxdfTable.size = setting->Size();
		bxdfTable.toc = new BxdfTocEntry[bxdfTable.size];
		host_bxdfs = new Bxdf[bxdfTable.size];
		bxdfcount = bxdfTable.size;

		// fill in the data
		int i = 0;
		for (Value::ConstValueIterator itr = setting->Begin(); itr != setting->End(); ++itr) {
			
			BxdfTocEntry e;
			Bxdf b;

			const Value& name = (*itr)["name"];
			e.bxdf_name = name.GetString();
			e.id = i;
			bxdfTable.toc[i] = e;

			const Value& type = (*itr)["type"];
			const char *temp = type.GetString();

			if (strcmp(temp, "NOHIT") == 0) b.type = crs::NOHIT;
			if (strcmp(temp, "NORMAL") == 0) b.type = crs::NORMAL;
			if (strcmp(temp, "LAMBERT") == 0) b.type = crs::LAMBERT;
			if (strcmp(temp, "OREN_NAYAR") == 0) b.type = crs::OREN_NAYAR;
			if (strcmp(temp, "CONDUCTOR") == 0) b.type = crs::CONDUCTOR;
			if (strcmp(temp, "DIELECTRIC") == 0) b.type = crs::DIELECTRIC;
			if (strcmp(temp, "EMISSION") == 0) b.type = crs::EMISSION;
			if (strcmp(temp, "SUBSURFACE") == 0) b.type = crs::SUBSURFACE;
			if (strcmp(temp, "CONSTANT") == 0) b.type = crs::CONSTANT;
			if (strcmp(temp, "SIMPLE_SKY") == 0) b.type = crs::SIMPLE_SKY;

			const Value& kd = (*itr)["albedo"];
			b.alb.x = kd[0].GetFloat();
			b.alb.y = kd[1].GetFloat();
			b.alb.z = kd[2].GetFloat();

			const Value& rpt = (*itr)["roughness"];
			b.rpt = rpt.GetFloat();

			const Value& ior = (*itr)["refraction"];
			b.ior = ior.GetFloat();

			host_bxdfs[i] = b;

			//cout << " Bxdf " << e.id << ", name: " << e.bxdf_name << ", type (id): " << temp << " (" << b.type << ")" << std::endl;
			//cout << " ->Kd: " << b.kd.x << "f, " << b.kd.y << "f, " << b.kd.z << "f" << std::endl;
			//cout << " ->Sh: " << b.sh << "f" << std::endl;

			i++;
		}

	}else{
		cout << " Error parsing scene file, no bxdfs found!" << std::endl;
		return EXIT_FAILURE;
	}

	if(dom.HasMember("spheres")){
		Value *setting;
		setting = Pointer("/spheres").Get(dom);
		assert(setting->IsArray());
		spherecount = setting->Size();
		cout << " " << spherecount << " Sphere(s) found..." << std::endl;

		// assign the memory
		host_spheres = new Sphere[spherecount];
		
		// fill in the data
		int i = 0;
		for (Value::ConstValueIterator itr = setting->Begin(); itr != setting->End(); ++itr){
			Sphere s;

			const Value& c = (*itr)["center"];
			s.center.x = c[0].GetFloat();
			s.center.y = c[1].GetFloat();
			s.center.z = c[2].GetFloat();

			const Value& r = (*itr)["radius"];
			s.radius = r.GetFloat();

			const Value& b = (*itr)["bxdf"];
			std::string name = b.GetString();
			s.bxdf = bxdfTable.getBxdfIdbyName(name);
			
			host_spheres[i] = s;

			cout << " Sphere " << i << ", bxdf id:" << s.bxdf << std::endl;

			i++;
		}

	}else{
		cout << " Error parsing scene file, no spheres found!" << std::endl;
		return EXIT_FAILURE;
	}

	// Device Camera
	cudaMalloc((void**)&device_camera, sizeof(crs::Camera));
	cudaMemcpy(device_camera, host_camera, sizeof(crs::Camera), cudaMemcpyHostToDevice);

	// Device Sphere
	cudaMalloc((void**)&device_spheres, sizeof(crs::Sphere)*spherecount);
	cudaMemcpy(device_spheres, host_spheres, sizeof(crs::Sphere)*spherecount, cudaMemcpyHostToDevice);

	// Device Bxdfs
	cudaMalloc((void**)&device_bxdfs, sizeof(crs::Bxdf)*bxdfcount);
	cudaMemcpy(device_bxdfs, host_bxdfs, sizeof(crs::Bxdf)*bxdfcount, cudaMemcpyHostToDevice);

	// assign the buffers on the host
	cc.SetupHostMemory();
	if(cc.state != CRS_NO_ERROR) cc.EvaluateState();

	// assign the buffers on the device
	cc.SetupDeviceMemory();
	if(cc.state != CRS_NO_ERROR) cc.EvaluateState();

	// --------------------------------------------------------
	// Start rendering
	// --------------------------------------------------------
	size_t start, end, elapsed;

	cc.dimension = k2D;
	cc.CalculateLaunchParamaters();

	// Prepare buffers
	crs::KERNEL_INIT <<<cc.gridSize, cc.blockSize>>>(cc.device_hitRecords, cc.device_pixels, cc.width, cc.height);
	cudaDeviceSynchronize();

	start = clock();
	// for each sample
	for (int i = 0; i < cc.samples; i++){

		crs::KERNEL_CAST_CAMERA_RAYS <<<cc.gridSize, cc.blockSize>>>(cc.device_hitRecords, device_camera, clock());
		cudaDeviceSynchronize();

		// for each bounce
		for (int j = 0; j < cc.depth; j++){

			crs::KERNEL_SPHEREINTERSECT <<<cc.gridSize, cc.blockSize>>>(device_spheres, spherecount, cc.device_hitRecords, cc.width, cc.height);
			cudaDeviceSynchronize();

			crs::KERNEL_BXDF <<<cc.gridSize, cc.blockSize >>>(device_bxdfs, cc.device_hitRecords, cc.device_pixels, cc.width, cc.height, cc.depth, clock());
			cudaDeviceSynchronize();
		}

		crs::KERNEL_ACCUMULATE<<<cc.gridSize, cc.blockSize >> >(cc.device_hitRecords, cc.device_pixels, cc.width, cc.height);
		cudaDeviceSynchronize();

		cout << "\r Rendered sample " << i + 1 << " of " << cc.samples;
	}
	end = clock();

	elapsed = end - start;
	float secs_elapsed = (float)elapsed / CLOCKS_PER_SEC;
	cout << "\n Rendering done! Finished in: "<< secs_elapsed << " seconds" << std::endl;
	cout << "--------------------------------------------------------------------------------------------- " << std::endl;

	// --------------------------------------------------------
	// End rendering
	// --------------------------------------------------------

	// Copy from device to host
	cc.CopyPixelBufferFromDeviceToHost();
	if(cc.state != CRS_NO_ERROR){
		cc.EvaluateState();
	}else{
		// save the file
		crs::SavePPM(cc.host_pixels, cc.width, cc.height, gamma_correction, output);
		cout << " Output saved to " << "output.ppm" << std::endl;
	}

	// Delete the host Camera
	if(host_camera != NULL){
		delete host_camera;
	}

	// Delete the device Camera
	if(device_camera != NULL){
		cudaFree(device_camera);
	}

	// Delete the host spheres
	if(host_spheres != NULL){
		delete[] host_spheres;
	}

	// Delete host bxdfs
	if (host_bxdfs != NULL) {
		delete[] host_bxdfs;
	}

	// Delete device bxdfs
	if (device_bxdfs != NULL) {
		cudaFree(device_bxdfs);
	}

	// Delete the device spheres
	if(device_spheres != NULL){
		cudaFree(device_spheres);
	}

	// Delete all memory
	cc.CleanupDevice();
	if(cc.state != CRS_NO_ERROR) cc.EvaluateState();

	cc.CleanupHost();
	if(cc.state != CRS_NO_ERROR) cc.EvaluateState();

	// reset the CUDA device
	cudaDeviceReset();

	cout << "--------------------------------------------------------------------------------------------- " << std::endl;
	cout << " The Crayon Rendering System says goodbye! " << std::endl;
	cout << "--------------------------------------------------------------------------------------------- " << std::endl;

	// get outta here
	return EXIT_SUCCESS;
}
