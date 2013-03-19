animation-character-identification
==================================

Implementation of a method for animation character identification from color images.

Installing on Windows with Visual Studio 2012
---------------------------------------------

These instructions were tested on Windows 7 64 bits. Visual Studio 2012 express is available at 
http://www.microsoft.com/visualstudio/eng/downloads , please download and install Visual Studio 2012 Express 
for Windows Desktop.

Requirements to build and run this program:
- CMake 2.6 or higher, available for download at http://www.cmake.org/cmake/resources/software.html
- The latest version of OpenCV, available for download at http://opencv.org/downloads.html
- The latest version of the Boost libraries, available for download at http://www.boost.org/users/download/

### Installing dependencies

If you have these programs already installed, please feel free to skip this section.

#### Installing CMake
Download the latest release of CMake from http://www.cmake.org/cmake/resources/software.html. 
Run the executable (cmake-2.x.xx-win32-x86.exe), and follow the installer instructions to install CMake.

#### Installing OpenCV
Download the latest release of OpenCV from http://opencv.org/downloads.html.
Detailed installation instructions for OpenCV with Visual Studio 10 (successfully tested with Visual Studio 2012) are
available at http://docs.opencv.org/doc/tutorials/introduction/windows_install/windows_install.html.

#### Installing Boost
Download the latest release of Boost from http://www.boost.org/users/download/. Extract the archive (boost_1_xx.zip)
to a folder of your choice. Go to this folder, then run bootstrap.bat . Once this is done, run the newly generated
b2.exe which will proceed to compile and install Boost.

### Installing
TODO
