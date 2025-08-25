

execute_process(
    COMMAND mkdir ${SRC_OUTPUT_PATH}
    COMMAND mkdir ${INCLUDE_OUTPUT_PATH}
    COMMAND protoc -I=${RIVER_PROTO_PATH} --cpp_out=${RIVER_PROTO_PATH} ${RIVER_PROTO_PATH}/AnalysisConfig.proto
    #COMMAND mv ${RIVER_PROTO_PATH}/AnalysisConfig.pb.h ${INCLUDE_OUTPUT_PATH}
    #COMMAND mv ${RIVER_PROTO_PATH}/AnalysisConfig.pb.cc ${SRC_OUTPUT_PATH}
)

file(RENAME ${RIVER_PROTO_PATH}/AnalysisConfig.pb.h ${INCLUDE_OUTPUT_PATH}/AnalysisConfig.pb.h)
file(RENAME ${RIVER_PROTO_PATH}/AnalysisConfig.pb.cc ${SRC_OUTPUT_PATH}/AnalysisConfig.pb.cc)
#file(PROTO_HEAD_FILE "&{RIVER_PROTO_PATH}/*.pb.h")
#file(PROTO_SRC_FILE "&{RIVER_PROTO_PATH}/*.pb.cc")

#foreach(fil ${PROTO_HEAD_FILE})
#    file(RENAME fil ${INCLUDE_OUTPUT_PATH})
#endforeach()

#foreach(fil ${PROTO_SRC_FILE})
#    file(RENAME fil ${SRC_OUTPUT_PATH})
#endforeach()
