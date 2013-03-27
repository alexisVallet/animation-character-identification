animation-character-identification
==================================

Implementation of a method for animation character identification from color images.

Compiling and running using Visual Studio
-----------------------------------------

Requires Visual Studio 2012, please download and install the the express version for windows desktop 
if you don't already have it installed on your computer: http://www.microsoft.com/visualstudio/eng/downloads

Download the latest sources from this url: https://github.com/alexisVallet/animation-character-identification/archive/master.zip

Extract the archive in the directory of your choice, then open 'aci/animation-character-identification.sln' with Visual Studio.
Compile the source code using "Generate the solution" in the menu. This produces the executable at 
aci/Debug/animation-character-identification.exe (or at aci/Release/ in release configuration).

Usage
-----

In the command line, the following segments the image and displays the resulting segmentation graph:

	animation-character-identification.exe image.png

And the following segments both images, then computes the tree walk kernel between the two segmentation graphs:

	animation-character-identification.exe image1.png image2.png

The program accepts any color image in a format recognized by OpenCV's imread (see OpenCV's documentation for details).
