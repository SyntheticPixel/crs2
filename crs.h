/*
 * crs.h
 *
 *  Created on: 02 Feb 2017
 *      Author: erik
 */

#pragma once
#ifndef CRS_H_
#define CRS_H_

/*
 * General include file for common CRS definitions and error handling
 */

// minimum available memory required, in bytes
#define MIN_REQUIRED_MEMORY (8*1024*1024)			/* 8 Mb required */

#define MIN_WIDTH 1
#define MIN_HEIGHT 1
#define MIN_SAMPLES 1
#define MAX_WIDTH 65535
#define MAX_HEIGHT 65535
#define MAX_SAMPLES 65535

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>

using namespace std;

namespace crs{
	typedef enum crs_status_enum{
		// CRAYON ERROR CODES
		CRS_NO_ERROR = 0x00,					/* no error */
		CRS_ABORT,								/* abort flag */
		CRS_NO_CUDA_DEVICE,							/* no cuda device found */
		CRS_NOT_ENOUGH_MEMORY,						/* not enough memory on device*/
		CRS_KERNEL_LAUNCH_ERROR,					/* launch kernel error */
		CRS_ERROR_HOST_MEMORY,						/* error assigning host memory */
		CRS_ERROR_DEVICE_MEMORY,					/* error assigning device memory */
		CRS_ERROR_HOST_CLEANUP,						/* error host cleaning */
		CRS_ERROR_DEVICE_CLEANUP,					/* error device cleaning */
		CRS_ERROR_COPY_FROM_DEVICE,					/* error copying memory from device */
		CRS_ERROR_UNKNOWN,							/* catch all error code */

	}STATUS;

}

#endif /* CRS_H_ */
