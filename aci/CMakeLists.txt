include_directories(${ACI_SOURCE_DIR}/aci)
link_directories(${ACI_BINARY_DIR}/aci)

configure_file (
  "${PROJECT_SOURCE_DIR}/aci/ACIConfig.h.in"
  "${PROJECT_BINARY_DIR}/aci/ACIConfig.h"
)

add_library(Utils Utils.cpp)
add_library(WeightedGraph WeightedGraph.cpp)
add_library(DisjointSet DisjointSet.cpp)
add_library(Felzenszwalb Felzenszwalb.cpp)
add_library(SegmentationGraph SegmentationGraph.cpp)
add_library(TreeWalkKernel TreeWalkKernel.cpp)
target_link_libraries(DisjointSet ${OpenCV_LIBS} Utils)
target_link_libraries(WeightedGraph ${OpenCV_LIBS} ${Boost_LIBS} Utils)
target_link_libraries(Felzenszwalb ${OpenCV_LIBS} WeightedGraph DisjointSet)
target_link_libraries(SegmentationGraph ${OpenCV_LIBS} WeightedGraph DisjointSet)
target_link_libraries(TreeWalkKernel ${OpenCV_LIBS} ${Boost_LIBS} DisjointSet WeightedGraph)

add_subdirectory(test)
