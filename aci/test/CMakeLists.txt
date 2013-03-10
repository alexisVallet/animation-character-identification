include_directories(${ACI_SOURCE_DIR}/aci/test)
include_directories(${ACI_BINARY_DIR}/aci/test)

add_executable(TestFelsenwalbExe Felsenwalb.cpp)
target_link_libraries(TestFelsenwalbExe ${OpenCV_LIBS} Felsenwalb)
add_test(TestFelsenwalb TestFelsenwalbExe)
