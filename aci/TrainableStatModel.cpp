#include "TrainableStatModel.h"

float TrainableStatModel::leaveOneOutCrossValidation(const Mat_<float> &samples, const Mat_<int> &classes) {
	int correctResults = 0;
	Mat_<int> sampleIdx(samples.rows - 1, 1);

	for (int i = 1; i < samples.rows; i++) {
		sampleIdx(i - 1, 0) = i;
	}

	for (int i = 0; i < samples.rows; i++) {
		this->clear();
		this->train(samples, classes, sampleIdx);
		int actual = (float)this->predict(samples.row(i));

		if (actual == classes(i,0)) {
			correctResults++;
		}

		//cout<<"actual = "<<actual<<", expected = "<<classes(i,0)<<endl;

		sampleIdx(i, 0) = i;
	}

	return (float)correctResults/(float)samples.rows;
}

BayesModel::BayesModel() {
}
    
CvStatModel *BayesModel::getStatModel() {
    return (CvStatModel*)&(this->internalStatModel);
}

void BayesModel::train(const Mat &trainData, const Mat &expectedResponses, const Mat &sampleIdx) {
    this->internalStatModel.train(trainData, expectedResponses, Mat(), sampleIdx);
}

float BayesModel::predict(const Mat &samples) {
    return this->internalStatModel.predict(samples);
}

void BayesModel::clear() {
    this->internalStatModel.clear();
}
    
KNearestModel::KNearestModel(int kValue) : k(kValue) {
}
    
CvStatModel *KNearestModel::getStatModel() {
    return (CvStatModel*)&(this->internalStatModel);
}
    
void KNearestModel::train(const Mat &trainData, const Mat &expectedResponses, const Mat &sampleIdx) {
    this->internalStatModel.train(trainData, expectedResponses, sampleIdx);
}
    
float KNearestModel::predict(const Mat &samples) {
	Mat results;
	Mat neighborResponses;
	Mat dists;

    float res = this->internalStatModel.find_nearest(samples, this->k, results, neighborResponses, dists);

	return res;
}

void KNearestModel::clear() {
    this->internalStatModel.clear();
}

ANNModel::ANNModel() {
}

ANNModel::ANNModel(Mat layerSizes, int activateFunc, double fparam1, double fparam2)
: internalStatModel(layerSizes, activateFunc, fparam1, fparam2), fparam2(fparam2)
{
}

CvStatModel *ANNModel::getStatModel() {
    return (CvStatModel*)&(this->internalStatModel);
}

void ANNModel::train(const Mat& trainData, const Mat& expectedResponses, const Mat &sampleIdx) {
    Mat layerSizes = this->internalStatModel.get_layer_sizes();
    int lastLayerSize = layerSizes.at<int>(0, layerSizes.cols-1);
    Mat responsesVectors = Mat::zeros(expectedResponses.rows, lastLayerSize, CV_32F);
    for (int i = 0; i < expectedResponses.rows; i++) {
        responsesVectors.at<float>(i, (int)expectedResponses.at<float>(i,0)) = 1;
    }
        
    this->internalStatModel.train(trainData, responsesVectors, Mat(), sampleIdx);
}

/**
 * Takes the value with highest probability.
 * 
 * @param samples
 * @return 
 */
float ANNModel::predict(const Mat& samples) {
    Mat networkOutput;
    this->internalStatModel.predict(samples, networkOutput);
    Point2i maxIndex;
    minMaxLoc(networkOutput, NULL, NULL, NULL, &maxIndex);
    return (float)maxIndex.x;
}

/**
 * Doesn't do anything, as calling the train method
 * clears the previous synaptic weights. 
 */
void ANNModel::clear() {
    
}
