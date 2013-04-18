animation-character-identification
==================================

Implementation of a method for animation character identification from color images.

Compiling and running using Visual Studio
-----------------------------------------

Requires Visual Studio 2010, please download and install the express version for windows desktop 
if you don't already have it installed on your computer: http://www.microsoft.com/visualstudio/eng/downloads . If you
already have another version of Visual Studio installed, please make sure to uninstall it beforehand as it may provoke
conflicts.

Requires the Windows SDK version 7.1, please download and install from http://www.microsoft.com/en-us/download/details.aspx?id=8279 .

Download the latest sources from this url: https://github.com/alexisVallet/animation-character-identification/archive/master.zip

Extract the archive in the directory of your choice, then open 'aci/animation-character-identification.sln' with Visual Studio.
Compile the source code using "Generate the solution" in the menu. This produces the executable at 
aci/Debug/animation-character-identification.exe (or at aci/Release/ in release configuration).

Usage
-----

In the command line, the following computes a recognition rate from the dataset in test/dataset using leave one out cross validation with various parameters:

	animation-character-identification.exe
