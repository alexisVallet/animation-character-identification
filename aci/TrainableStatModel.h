/** @file */
/* 
 * File:   TrainableStatModel.h
 * Author: Alexis
 *
 * This file was originally written by myself for my previous project.
 *
 * Created on 1 décembre 2012, 11:15
 */

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Wrapper around CvStatModels which implement a train and predict (-like)
 * method.
 */
class TrainableStatModel {
public:
    /**
     * Returns the opencv statistical model which this statistical model
     * is based on.
     * 
     * @return the opencv statistical model which this statistical model
     * is based on.
     */
    virtual CvStatModel *getStatModel() = 0;
    /**
     * Trains the classifier using a specific training base.
     * 
     * @param trainData the input training data.
     * @param expectedResponses the expected response to the training data.
     */
    virtual void train(const Mat &trainData, const Mat &expectedResponses, const Mat &sampleIdx) = 0;
    /**
     * Predicts the class of a sample.
     * 
     * @param samples one or many samples to compute the class for.
     * @return the predicted class of the sample.
     */
    virtual float predict(const Mat &samples) = 0;
    /**
     * Clears the training of the internal statistical model.
     */
    virtual void clear() = 0;

	/**
	 * Computes the leave one out cross validation recognition rate given
	 * a matrix of row samples and a column matrix of their corresponding class.
	 *
	 * @param samples matrix of row samples to compute the recognition rate from.
	 * @param classes column vector associating to each sample its corresponding class.
	 * @return a rate in the [0;1] range indicating the ratio of correctly classified
	 * samples by training with all the others.
	 */
	float leaveOneOutCrossValidation(const Mat_<float> &samples, const Mat_<int> &classes);
};

/**
 * Wrapper class for Bayes's classifier.
 */
class BayesModel : public TrainableStatModel {
private:
    CvNormalBayesClassifier internalStatModel;

public:    
    BayesModel();
        
    CvStatModel *getStatModel();

    void train(const Mat &trainData, const Mat &expectedResponses, const Mat &sampleIdx);

    float predict(const Mat &samples);
    
    void clear();
};

#define DEFAULT_K 1

/**
 * Wrapper class around the K nearest neighbors classifier.
 */
class KNearestModel : public TrainableStatModel {
private:
    int k;
    CvKNearest internalStatModel;
    
public:
    
    KNearestModel(int kValue = DEFAULT_K);
    
    CvStatModel *getStatModel();
    
    void train(const Mat &trainData, const Mat &expectedResponses, const Mat &sampleIdx);
    
    float predict(const Mat &samples);
    
    void clear();
};

/**
 * Wrapper class around the artificial neural network classifier.
 */
class ANNModel : public TrainableStatModel {
private:
    CvANN_MLP internalStatModel;
    double fparam2;
    
public:
    ANNModel();
    
    ANNModel(Mat layersSize, int activateFunc = CvANN_MLP::SIGMOID_SYM, double fparam1 = 1, double fparam2 = 1);
    
    CvStatModel *getStatModel();
    
    void train(const Mat &trainData, const Mat &expectedResponses, const Mat &sampleIdx);
    
    float predict(const Mat &samples);
    
    void clear();
};

/**
 * Wrapper class around the C-SVM classifier.
 */
class SVMModel : public TrainableStatModel {
private:
	CvSVM internalStatModel;

public:
	SVMModel();

	CvStatModel *getStatModel();

	void train(const Mat &trainData, const Mat &expectedResponses, const Mat &sampleIdx);

	float predict(const Mat &samples);

	void clear();
};