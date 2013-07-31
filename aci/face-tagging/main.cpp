#include "../Utils.hpp"

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace Eigen;

#define NB_IMAGE_PER_CHAR 5

void mouse(int ev, int x, int y, int, void *p) {
	Vector2d *position = (Vector2d*)p;

	if (ev != EVENT_LBUTTONDOWN) {
		return;
	}

	(*position) = Vector2d((double)x,(double)y);
	cout<<"new position "<<(*position)<<endl;
}

/**
 * Simple utility program going through the images in the dataset, prompting
 * the user to click on the position of the face. Writes the results to a .csv
 * file.
 */
int main(int argc, char **argv) {
	if (argc < 2) {
		cout<<"Please give a folder name as argument"<<endl;
	} else {
		char *folderName = argv[1];

		cout<<"loading dataset..."<<endl;
		char *charaNames[] = {"rufy", "ray", "miku", "majin", "lupin", "kouji", "jigen", "conan", "chirno", "char", "asuka", "amuro", NULL};
		vector<pair<Mat_<Vec3b>, Mat_<float> > > dataSet;
		Mat_<int> classes;

		loadDataSet("../test/dataset/", charaNames, NB_IMAGE_PER_CHAR, dataSet, classes);

		vector<pair<string, Vector2d>, aligned_allocator<pair<string, Vector2d> > > faceData;
		faceData.reserve(dataSet.size());

		// prompting the user for face positions
		for (int i = 0; charaNames[i] != NULL; i++) {
			for (int j = 0; j < NB_IMAGE_PER_CHAR; j++) {
				char suffix[] = {'_', 'a' + j, '\0'};
				Vector2d facePosition(0,0);
				
				namedWindow("face");
				setMouseCallback("face", mouse, &facePosition);
				imshow("face", dataSet[j + NB_IMAGE_PER_CHAR * i].first);
				waitKey(0);
				
				stringstream nameStream;

				nameStream<<charaNames[i]<<suffix;

				faceData.push_back(pair<string,Vector2d>(nameStream.str(),facePosition));
			}
		}

		// writing results to a csv file
		ofstream faceDataFile;
		stringstream faceDataFilenameStream;

		faceDataFilenameStream<<folderName<<"faceData.csv"<<endl;

		faceDataFile.open(faceDataFilenameStream.str());

		for (int i = 0; i < dataSet.size(); i++) {
			cout<<
				faceData[i].first<<", "<<
				faceData[i].second(0)<<", "<<
				faceData[i].second(1)<<endl;
		}
		
		faceDataFile.close();
	}
}
