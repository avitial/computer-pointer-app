#!/bin/bash
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6

exec 1>output/stdout.log 2>output/stderr.log

mkdir -p output


INPUT=$1
PRECISION=$2
DEVICE=$3
API_TYPE=$4
LOG_FILE="${DEVICE}_${PRECISION}_${API_TYPE}.txt"



# Run python app script
python3 main.py \
    -m_fd intel/face-detection-adas-0001/${PRECISION}/face-detection-adas-0001.xml \
    -m_fl intel/landmarks-regression-retail-0009/${PRECISION}/landmarks-regression-retail-0009.xml \
    -m_hpe intel/head-pose-estimation-adas-0001/${PRECISION}/head-pose-estimation-adas-0001.xml \
    -m_ge intel/gaze-estimation-adas-0002/${PRECISION}/gaze-estimation-adas-0002.xml \
    -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so \
    -i ${INPUT} \
    -d ${DEVICE} \
    -api ${API_TYPE} \
    -sb true > output/${LOG_FILE}

cd output

tar zcvf output.tgz stdout.log stderr.log
