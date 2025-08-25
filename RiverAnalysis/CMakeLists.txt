cmake_minimum_required(VERSION 3.5)

project(RiverAnalysis)
message(STATUS "Compiling")

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -W -std=c++11")

add_definitions(-D _DEBUG)

#add_definitions(-DANALYSIS_SHOW)
#add_definitions(-DTENSORFLOW_THREAD_SHOW)
#add_definitions(-DFLOATER_THREAD_SHOW)
#add_definitions(-DWATERCOLOR_THREAD_SHOW)
#add_definitions(-DLITTER_THREAD_SHOW)
#add_definitions(-DINVADE_THREAD_SHOW)
# 三方库
find_package(OpenCV 3.2.0)
if(OpenCV_FOUND)
    message(STATUS "OpenCV is Found")
else()
    message(FATAL_ERROR "OpenCV is Not Found")
endif()

find_package(Protobuf REQUIRED)
if(PROTOBUF_FOUND)
   message(STATUS "Protobuf is Found")
else()
   message(FATAL_ERROR "Protobuf is Not Found")
endif()


set(Protobuf_INCLUDE_DIRS "/usr/local/protobuf/include")
set(Protobuf_LIBRARIES_DIRS "/usr/local/protobuf/lib")
include_directories(${Protobuf_INCLUDE_DIRS})
link_directories(${Protobuf_LIBRARIES_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(TENSORFLOW_LIB_PATH "./libtensorflow")
include_directories(    
 ./tensorflow-1.11.0
 ./tensorflow-1.11.0/bazel-genfiles
 ./tensorflow-1.11.0/tensorflow/contrib/makefile/downloads/nsync/public
 ./tensorflow-1.11.0/bazel-bin/tensorflow
 ./tensorflow-1.11.0/tensorflow/contrib/makefile/downloads/absl
 ./tensorflow-1.11.0/tensorflow/contrib/makefile
  /usr/local/include/eigen3
  /usr/local/include/eigen3/unsupported
)
link_directories(${TENSORFLOW_LIB_PATH})

# 输出路径
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/output/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/output/lib")
set(INCLUDE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/output/include")
set(SRC_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/output/src")
# 本地文件路径
set(RIVER_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/include")
set(RIVER_SRC_PATH "${PROJECT_SOURCE_DIR}/src")
set(RIVER_TEST_PATH "${PROJECT_SOURCE_DIR}/test")
set(RIVER_PROTO_PATH "${PROJECT_SOURCE_DIR}/proto")
add_subdirectory(proto)

file(GLOB PROTO_SRC_FILES "${SRC_OUTPUT_PATH}/*.pb.cc")

# 本地文件
include_directories(/usr/local/include/freetype2)
include_directories(
    ${RIVER_INCLUDE_PATH}
    ${INCLUDE_OUTPUT_PATH}
    ${PROJECT_SOURCE_DIR}/proto
    /usr/local/openssl/include
)

set(RIVER_ANALYSISI_LIB_FILE 
    ${RIVER_SRC_PATH}/DefineColor.cpp
    ${RIVER_SRC_PATH}/RiverThread.cpp
    ${RIVER_SRC_PATH}/Analysis.cpp
    ${RIVER_SRC_PATH}/AnalysisManager.cpp

    ${RIVER_SRC_PATH}/TensorflowThread.cpp
    ${RIVER_SRC_PATH}/GaussBgModeling.cpp
    ${RIVER_SRC_PATH}/WaterColorThread.cpp
    ${RIVER_SRC_PATH}/FloaterThread.cpp
    ${RIVER_SRC_PATH}/LitterThread.cpp
    ${RIVER_SRC_PATH}/InvadeThread.cpp
    ${RIVER_SRC_PATH}/FishingThread.cpp
    ${RIVER_SRC_PATH}/WaterGaugeThread.cpp

    ${RIVER_SRC_PATH}/cv320PutChText.cpp
    ${RIVER_SRC_PATH}/mongoose.c
    ${RIVER_SRC_PATH}/cJSON.c
    ${RIVER_SRC_PATH}/AlarmMsgQueue.cpp
    ${RIVER_SRC_PATH}/http_client.cpp


    ${RIVER_SRC_PATH}/AnalysisConfig.pb.cc
)

add_library(Analysis_static STATIC ${RIVER_ANALYSISI_LIB_FILE})
#add_library(Analysis SHARED ${RIVER_ANALYSISI_LIB_FILE})

link_directories(/usr/local/openssl/lib)
link_directories(${LIBRARY_OUTPUT_PATH})
#set_target_properties(Analysis_static PROPERTIES OUTPUT_NAME "Analysis")

add_executable(RiverAnalysis ${PROJECT_SOURCE_DIR}/RiverAnalysis.cpp)

target_link_libraries(RiverAnalysis
    Analysis_static
    tensorflow_cc
    tensorflow_framework
    freetype
    X11
    pthread
    glog
    gflags
    ssl
    crypto
    ${OpenCV_LIBS}
    ${Protobuf_LIBRARIES}
)
