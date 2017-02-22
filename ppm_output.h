/*
 * ppm_output.h
 *
 *  Created on: 02 Feb 2017
 *      Author: erik
 */

#ifndef PPM_OUTPUT_H_
#define PPM_OUTPUT_H_

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "crs.h"
#include "pixelbuffer.h"

using namespace std;
using namespace glm;
using namespace crs;

namespace crs{
	/*
	@brief
	Convert a float (0.0 < f < 1.0) to unsigned char
	*/
	unsigned char FloatToShort(float f, float g){
		// clamp between 0.0 and 1.0
		float v = std::fmax(0.0f, std::fmin(f, 1.0f));

		// gamma correct
		float c = pow(v , (1.0f/g));

		// convert float to char
		return (unsigned char)(c*255.99f);
	}

	/*
	@brief
	Save to 8-bit ppm file
	params: data, width, height, filename
	*/
	void SavePPM(PixelBuffer* data, unsigned int w, unsigned int h, float gamma, std::string filename){

		if(data == NULL){
			cout << " Invalid data... no file saved" << "\n";
			return;
		}

		ofstream outputFile;
		outputFile.open(filename, ofstream::out | ofstream::binary);

		// PPM header, P6 = binary
		outputFile << "P6\n" << w << " " << h << "\n" << "255" << "\n";

		unsigned char r, g, b;

		// loop over each pixel
		long  i;
		long  j;
		j = (w*h);

		// image write
		for (i = 0; i < j; i++) {
			PixelBuffer a;
			a = data[i];

			vec3 t = a.getAverage();

			r = FloatToShort(t.x, gamma);
			g = FloatToShort(t.y, gamma);
			b = FloatToShort(t.z, gamma);
			outputFile << r << g << b;
		}

		// Close the file
		outputFile.close();

	}

}

#endif /* PPM_OUTPUT_H_ */
