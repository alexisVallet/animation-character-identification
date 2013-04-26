#include "KuwaharaFilter.h"


					  
class lab : Vec3f{
	double operator-(lab& lab){
	
	};
};

template <typename T>
T matAverage(Mat_<T> &mat){

	T sum;

	for(unsigned int i = 0; i < mat.rows ;i++){
		for(unsigned int j = 0; j < mat.cols ;j++){
			sum += mat(i,j);
		}
	}

	int  elements_num = (mat.rows * mat.cols);

	//Range of L*a*b* is not knowing...
	//assert(average[0] >= 0 && average[0] < 256 && average[1] >= 0 && average[1] < 256 && average[2] >= 0 && average[2] < 256 );

	return  sum / elements_num;
}

float matLabVariance(Mat_<Vec3f> &mat, Vec3f average){
	
	float sum = 0;
	
	for(unsigned int i = 0; i < mat.rows ;i++){
		for(unsigned int j = 0; j < mat.cols ;j++){
			sum += sqrt( pow(average[0] - mat(i,j)[0],2) + pow(average[1] - mat(i,j)[1],2) +  pow(average[2] - mat(i,j)[2],2));
		}
	}

	return sum / (mat.rows * mat.cols);
}

void KuwaharaFilter(Mat_<Vec<uchar,3>> &rgbSrc, Mat_<Vec<uchar,3>> &dest, uchar filterSize){

	assert(filterSize % 2 == 1);

	Mat_<Vec3f> labSrc(rgbSrc.rows,rgbSrc.cols), labDest(rgbSrc.rows,rgbSrc.cols);
	Mat_<Vec3f> rgb3f = rgbSrc / 255.;

	cvtColor(rgb3f, labSrc,  CV_RGB2Lab);



	Vec3f min_average;
	float min_variance;

	int halfSize = filterSize / 2;

	for(int i = 0; i < labSrc.rows ; i++){
		for(int j = 0; j < labSrc.cols ; j++){							 
	
			/*
			* Should take into account if variances are same in multiple regions.
			*/
			//cout<<"i:"<< i <<" j:"<< j << endl;
			//cout<<"halfSize:"<< halfSize << endl;

			min_variance = FLT_MAX;

			// top left region
			if((i - halfSize) >= 0 && (j - halfSize) >= 0){ 
				
				Mat_ <Vec3f> croppedSrc = labSrc.rowRange(i - halfSize, i).colRange(j - halfSize, j);
				Vec3f average = matAverage<Vec3f>(croppedSrc);
				float variance = matLabVariance(croppedSrc,average);

				if(variance < min_variance){
					min_variance = variance;
					min_average = average;
				}
			}

			// top right region
			if((i + halfSize) < labSrc.rows && (j - halfSize) >= 0){ 
				
				Mat_ <Vec3f> croppedSrc = labSrc.rowRange(i, i + halfSize ).colRange(j - halfSize, j);
				Vec3f average = matAverage<Vec3f>(croppedSrc);
				float variance = matLabVariance(croppedSrc,average);

				if(variance < min_variance){
					min_variance = variance;
					min_average = average;
				}
			}
			
			// bottom left region
			if((i - halfSize) >= 0 && (j + halfSize) < labSrc.cols){ 
				
				Mat_ <Vec3f> croppedSrc = labSrc.rowRange(i - halfSize, i).colRange(j , j + halfSize);
				Vec3f average = matAverage<Vec3f>(croppedSrc);
				float variance = matLabVariance(croppedSrc,average);

				if(variance < min_variance){
					min_variance = variance;
					min_average = average;
				}
			}

			// bottom right region
			if((i + halfSize) < labSrc.rows && (j + halfSize) < labSrc.cols){ 
				
				Mat_ <Vec3f> croppedSrc = labSrc.rowRange(i, i + halfSize ).colRange(j, j + halfSize);
				Vec3f average = matAverage<Vec3f>(croppedSrc);
				float variance = matLabVariance(croppedSrc,average);

				if(variance < min_variance){
					min_variance = variance;
					min_average = average;
				}
			}

			//cout<<" checked. " << endl;
			labDest(i,j) = min_average;
			
		}
	}
	cvtColor(labDest,rgb3f,CV_Lab2RGB);
	rgb3f = rgb3f * 255;

	dest = Mat_<Vec3b>(rgb3f);

}