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
	unsigned char FloatToShort(float f){
		return (unsigned char)(f*255.0f);
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

			// gamma correct
			vec3 c;
			c.x = pow(t.x , 1.0f/gamma);
			c.y = pow(t.y , 1.0f/gamma);
			c.z = pow(t.z , 1.0f/gamma);

			r = FloatToShort(c.x);
			g = FloatToShort(c.y);
			b = FloatToShort(c.z);
			outputFile << r << g << b;
		}

		// Close the file
		outputFile.close();

	}

}

#endif /* PPM_OUTPUT_H_ */
