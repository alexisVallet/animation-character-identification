#include "main.h"

int main(int argc, char** argv) {
	time_t currentTime = time(0);
	stringstream filenameStream;

	filenameStream<<"../stats/results.csv"<<endl;
	
	ofstream statFile;
	statFile.open(filenameStream.str());
	parameterTuning(statFile);
	statFile.close();

	return 0;
}
