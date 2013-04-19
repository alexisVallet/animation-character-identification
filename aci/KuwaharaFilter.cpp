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

float MatRGBvariance(Mat_<Vec<uchar,3>> &image, Vec<uchar,3> averageColor, unsigned int x_start,unsigned int x_end,unsigned int y_start,unsigned int y_end){
	
	Vec<float,3> HSV,averageHSV;

	cvtColor(averageColor,averageHSV,CV_RGB2HSV);
	float sum = 0;
	
	for(unsigned int i = x_start; i <= x_end ;i++){
		for(unsigned int j = y_start; j <= y_end ;j++){
			cvtColor(image(i,j),HSV,CV_RGB2HSV);
			sum += pow((HSV[2] - averageHSV[2]),2);
		}
	}

	unsigned int pixel_num = ((x_end - x_start) * (y_end - y_start));

	return sum / pixel_num;
}

void KuwaharaFilter(Mat_<Vec<uchar,3>> &src, Mat_<Vec<uchar,3>> &dest, uchar filterSize){

	assert(filterSize % 2 == 0);

	Mat_<Vec<uchar,3>> Filtered(src.rows, src.cols);
	Vec<uchar,3> min_averageRGB;
	float min_variance = 65536;
	uchar halfSize = filterSize / 2;

	for(unsigned int i = 0; i < src.rows ; i++){
		for(unsigned int j = 0; i < dest.cols ; j++){							 
	
			/*
			* Should take into account if variances are same in multiple regions.
			*/

			// top left region
			if((i - halfSize) >= 0 && (j - halfSize) >= 0){ 
				Vec<uchar,3> averageRGB = MatRGBaverage(src, i - halfSize, i, j - halfSize, j);
				float variance = MatRGBvariance(src,averageRGB,i - halfSize, i, j - halfSize, j);
				if(variance < min_variance){
					min_variance = variance;
					min_averageRGB = averageRGB;
				}
			}

			// top right region
			if((i + halfSize) < src.rows && (j - halfSize) >= 0){
				Vec<uchar,3> averageRGB = MatRGBaverage(src, i , i + halfSize, j - halfSize, j);
				float variance = MatRGBvariance(src,averageRGB,i , i + halfSize, j - halfSize, j);
				if(variance < min_variance){
					min_variance = variance;
					min_averageRGB = averageRGB;
				}
			}

			// bottom left region
			if((i - halfSize) >= 0 && (j + halfSize) < src.cols){ 
				Vec<uchar,3> averageRGB = MatRGBaverage(src, i - halfSize, i, j , j + halfSize);
				float variance = MatRGBvariance(src,averageRGB,i - halfSize, i, j , j + halfSize);
				if(variance < min_variance){
					min_variance = variance;
					min_averageRGB = averageRGB;
				}
			}

			// bottom right region
			if((i + halfSize) < src.rows && (j + halfSize) < src.cols){ 
				Vec<uchar,3> averageRGB = MatRGBaverage(src, i , i + halfSize, j, j + halfSize);
				float variance = MatRGBvariance(src,averageRGB,i , i + halfSize, j, j + halfSize);
				if(variance < min_variance){
					min_variance = variance;
					min_averageRGB = averageRGB;
				}
			}

			dest(i,j) = min_averageRGB;

		}
	}
}