/*
 * main.h
 *
 *  Created on: 02 Feb 2017
 *      Author: erik
 */

#pragma once
#ifndef MAIN_H_
#define MAIN_H_

// Includes
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include "rapidjson/pointer.h"
#include <rapidjson/istreamwrapper.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

#include "crs.h"
#include "camera.h"
#include "cuda_context.h"
#include "pixelbuffer.h"
#include "rand.h"
#include "ray.h"
#include "hit_record.h"
#include "sphere.h"
#include "ppm_output.h"
#include "bxdf.h"

#endif /* MAIN_H_ */
