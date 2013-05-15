#include "main.h"

int main(int argc, char** argv) {
	time_t currentTime = time(0);
	struct tm *now = localtime(&currentTime);
	stringstream filenameStream;

	filenameStream<<"../stats/results.csv"<<endl;

	cout<<filenameStream.str()<<endl;
	
	ofstream statFile;
	statFile.open(filenameStream.str());
	parameterTuning(statFile);
	statFile.close();

	return 0;
}
