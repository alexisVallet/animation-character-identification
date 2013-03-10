include_directories(${ACI_SOURCE_DIR}/aci)
link_directories(${ACI_BINARY_DIR}/aci)

configure_file (
  "${PROJECT_SOURCE_DIR}/aci/ACIConfig.h.in"
  "${PROJECT_BINARY_DIR}/aci/ACIConfig.h"
)

add_library(Utils Utils.cpp)
add_library(WeightedGraph WeightedGraph.cpp)
add_library(DisjointSet DisjointSet.cpp)
add_library(Felsenwalb Felsenwalb.cpp)
target_link_libraries(DisjointSet ${OpenCV_LIBS} Utils)
target_link_libraries(WeightedGraph Utils)
target_link_libraries(Felsenwalb ${OpenCV_LIBS} WeightedGraph DisjointSet)

add_subdirectory(test)