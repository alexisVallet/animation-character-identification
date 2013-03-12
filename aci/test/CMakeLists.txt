include_directories(${ACI_SOURCE_DIR}/aci/test)
include_directories(${ACI_BINARY_DIR}/aci/test)

add_executable(TestFelzenszwalbExe Felzenszwalb.cpp)
target_link_libraries(TestFelzenszwalbExe ${OpenCV_LIBS} Felzenszwalb)
add_test(TestFelzenszwalb TestFelzenszwalbExe)

add_executable(TestDisjointSetExe DisjointSet.cpp)
target_link_libraries(TestDisjointSetExe DisjointSet)
add_test(TestDisjointSet TestDisjointSetExe)

add_executable(TestSegmentationGraphExe SegmentationGraph.cpp)
target_link_libraries(TestSegmentationGraphExe ${OpenCV_LIBS} SegmentationGraph Felzenszwalb)
add_test(TestSegmentationGraph TestSegmentationGraphExe)