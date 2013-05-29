//
//  ContourDetectionforAnime.cpp
//  animation-charactor-identification for Mac
//
//  Created by Yuki NAKAGAWA on 2013/05/07.
//  Copyright (c) 2013年 Yuki & Alexis. All rights reserved.
//

#include "ContourDetectionforAnime.h"

enum pixelIsContourOrNot { Not_Contour, Candidate, Contour };
enum trackingState { Undefined, Raised_HighToHigh, Raised_LowToLow };

// In each the laplasian tracking step, value raised up high negative to high positive and
// falling down from high positive to high negative within "maxContourWidth", then pixels bitween
// Raising and Falling are may "Contour".

void ContourDetectionforAnime(Mat_<Vec3b> &src, float threshold_a , float threshold_b ,int maxContourWidth){
    Mat_<uchar> gray,dest,cand(src.rows,src.cols);
    Mat_<float> gray_float,laplacianed;
    Mat_<pixelIsContourOrNot> pixel_state(src.rows,src.cols);
    pixel_state = Mat_<pixelIsContourOrNot>::zeros(src.rows,src.cols);
    cand = Mat_<uchar>::zeros(src.rows,src.cols);
    
    //Same style as Canny 
    float threshold_high, threshold_low;
    if(threshold_a > threshold_b){
        threshold_high = threshold_a;
        threshold_low = threshold_b;
    }else{
        threshold_high = threshold_b;
        threshold_low = threshold_a;
    }
    
    cvtColor(src,gray,CV_RGB2GRAY);
    gray_float = Mat_<float>(gray) / 255.0;
    dest = Mat::zeros(src.rows, src.cols, CV_32S);
    Laplacian(gray_float, laplacianed, CV_32F, 3);
    
    stack<Vec2i> contourStack;
    
    trackingState state;

//---------------------------------------------------------------------------------------------

    for(int i = 0 ; i < laplacianed.rows ; i++ ){

        state = Undefined;
        int pixelsFromLastRaising = 0;
        
        for(int j = 1 ; j < laplacianed.cols ; j++ ){
            
            if(pixelsFromLastRaising > maxContourWidth){        //When pixels number from last raised up is over the MaxLineWidth, these pixels are not Contour.
                pixelsFromLastRaising = 0;
                state = Undefined;
            }
            
            if(laplacianed(i,j-1) < -threshold_high && laplacianed(i,j) > threshold_high){          // Raise with High Threshold
                state = Raised_HighToHigh;
                pixelsFromLastRaising = 0;
                
            }else if(laplacianed(i,j-1) < -threshold_low && laplacianed(i,j) > threshold_low){      // Raise with Low Threshold
                state = Raised_LowToLow;
                pixelsFromLastRaising = 0;
                
            }else if(laplacianed(i,j-1) > threshold_high && laplacianed(i,j) < -threshold_high){    // Fall with High Threshold
                if(state == Raised_HighToHigh){                                                     // High to High
                    for(int k = j - pixelsFromLastRaising; k < j; k++ ){
                        dest(i,k) = 255;
                        pixel_state(i,k) = Contour;
                        contourStack.push(Vec2i(i,k));
                    }
                }else{                                                                              // Low to Low
                    for(int k = j - pixelsFromLastRaising; k < j; k++ ){
                        cand(i,k) = 255;
                        pixel_state(i,k) = Candidate;
                    }
                }
                state = Undefined;
            }else if(laplacianed(i,j-1) > threshold_low && laplacianed(i,j) < -threshold_low){      //High/Low To Low (These pixels are Always Candidates in This Case)
                for(int k = j - pixelsFromLastRaising; k < j; k++ ){
                    cand(i,k) = 255;
                    pixel_state(i,k) = Candidate;
                }
                state = Undefined;
            }
        
            if(state != Undefined ){
                pixelsFromLastRaising++;
            }
        }
    }
    
    //---------------------------------------------------------------------------------------------
    
    for(int j = 0 ; j < laplacianed.cols ; j++ ){
        
        state = Undefined;
        int pixelsFromLastRaising = 0;
        
        for(int i = 1 ;  i < laplacianed.rows ; i++ ){
            
            if(pixelsFromLastRaising > maxContourWidth){        //When pixels number from last raised up is over the MaxLineWidth, these pixels are not Contour.
                pixelsFromLastRaising = 0;
                state = Undefined;
            }
            
            if(laplacianed(i-1,j) < -threshold_high && laplacianed(i,j) > threshold_high){          // Raise with High Threshold
                state = Raised_HighToHigh;
                pixelsFromLastRaising = 0;
                
            }else if(laplacianed(i-1,j) < -threshold_low && laplacianed(i,j) > threshold_low){      // Raise with Low Threshold
                state = Raised_LowToLow;
                pixelsFromLastRaising = 0;
                
            }else if(laplacianed(i-1,j) > threshold_high && laplacianed(i,j) < -threshold_high){    // Fall with High Threshold
                if(state == Raised_HighToHigh){                                                     // High to High
                    for(int k = i - pixelsFromLastRaising; k < i; k++ ){
                        dest(k,j) = 255;
                        pixel_state(k,j) = Contour;
                        contourStack.push(Vec2i(k,j));
                    }
                }else{                                                                              // Low to Low
                    for(int k = i - pixelsFromLastRaising; k < i; k++ ){
                        cand(k,j) = 255;
                        
                        if (pixel_state(k,j) != Contour ){ pixel_state(k,j) = Candidate;}
                    }
                }
                state = Undefined;
            }else if(laplacianed(i-1,j) > threshold_low && laplacianed(i,j) < -threshold_low){      //High/Low To Low (These pixels are Always Candidates in This Case)
                for(int k = i - pixelsFromLastRaising; k < i; k++ ){
                    cand(k,j) = 255;
                    if (pixel_state(k,j) != Contour ){ pixel_state(k,j) = Candidate;}
                }
                state = Undefined;
            }
            
            if(state != Undefined ){
                pixelsFromLastRaising++;
            }
        }
    }
  
    //---------------------------------------------------------------------------------------------
    
    
    while(!contourStack.empty()){
        //TODO : 線を繋げる処理を書く
        Vec2i v = contourStack.top();
        if( pixel_state(v) == Candidate ){
            pixel_state(v) = Contour;
            contourStack.push(v + Vec2i( 1, 0));
            contourStack.push(v + Vec2i(-1, 0));
            contourStack.push(v + Vec2i( 0, 1));
            contourStack.push(v + Vec2i( 0,-1));
            contourStack.push(v + Vec2i( 1,-1));
            contourStack.push(v + Vec2i( 1, 1));
            contourStack.push(v + Vec2i(-1,-1));
            contourStack.push(v + Vec2i(-1, 1));
            
            dest(v) = 255;
        }else{
            pixel_state(v) = Not_Contour;
            
            cand(v) = 0;
        };
        contourStack.pop();
    }
    
    //---------------------------------------------------------------------------------------------
    
    

    vector<Mat_<uchar>> img;
    
    img.push_back(dest);
    img.push_back(Mat_<uchar>::zeros(src.rows, src.cols));
    img.push_back(cand);
    
    Mat_<Vec3b> image;
    merge(img,image);
    
    imshow("gray", gray);
    imshow("Lapracianed", laplacianed+ 0.5);
    imshow("line detected", image);
    //imshow("line Connected", pixel_state * 127);
    
}