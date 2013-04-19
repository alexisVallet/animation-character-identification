#include "KuwaharaFilter.h"

Vec<uchar,3> MatRGBaverage(Mat_<Vec<uchar,3>> &image, unsigned int x_start,unsigned int x_end,unsigned int y_start,unsigned int y_end){
	
	assert(x_start < x_end && y_start < y_end);
	assert(x_start > 0 && x_end > 0 && y_start > 0 && y_end > 0);

	Vec<float,3> sum;

	for(unsigned int i = x_start; i <= x_end ;i++){
		for(unsigned int j = y_start; j <= y_end ;j++){
			sum += image(i,j);
		}
	}

    unsigned int pixel_num = ((x_end - x_start) * (y_end - y_start));
	
	Vec<uchar,3> average(sum[0] / pixel_num,
						sum[1] / pixel_num,
						sum[2] / pixel_num);

	assert(average[0] >= 0 && average[0] < 256 && average[1] >= 0 && average[1] < 256 && average[2] >= 0 && average[2] < 256 );

	return average;
}

Vec<uchar,3> MatRGBvariance(Mat_<Vec<uchar,3>> &image, Vec<uchar,3> averageColor){
	
	cv_RG
	
	for(unsigned int i = x_start; i <= x_end ;i++){
		for(unsigned int j = y_start; j <= y_end ;j++){
			sum += image(i,j);
		}
	}

    unsigned int pixel_num = ((x_end - x_start) * (y_end - y_start));
	
	Vec<uchar,3> average(sum[0] / pixel_num,
						sum[1] / pixel_num,
						sum[2] / pixel_num);

	assert(average[0] >= 0 && average[0] < 256 && average[1] >= 0 && average[1] < 256 && average[2] >= 0 && average[2] < 256 );

	return average;
}

Mat_<Vec<uchar,3>> KuwaharaFilter(Mat_<Vec<uchar,3>> &image, uchar filterSize){

	assert(filterSize % 2 == 0);

	uchar halfSize = filterSize / 2;

	for(unsigned int i = 0; i < image.rows ; i++){
		for(unsigned int j = 0; i < image.cols ; j++){
			

		}
	}
	return(image);
}