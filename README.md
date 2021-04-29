# Computer Pointer Controller
In this project we have used a Gaze Detection model to control the mouse pointer of a computer. The purpose of this project was to demonstrate the ability to run multiple models in the same machine and coordinate the flow of data between such models.

When the application starts, a window displays the input video user selected. If a face is detected in the video it, it estimates the head/eyes position and then moves the mouse cursor based on eyes and head movement. For the application to function properly a single person's face must be in the frame, otherwise it will wait.

The mouse controller settings are preset for fast and high-precision movement. A feature to increase/decrease UI responsiveness has been added to the application for better feel, and its with the center value by default for faster UI feel. This setting can be set using `-s, --speed` parameter with values in range of [1-30].

The video feed is shown by default without any inference results from intermediate models. These can be enabled using the `-v, --visualize` parameter or at runtime with Space key with focus on output window. Note this may affect the performance of the application.

## Project Set Up and Installation
This project has been developed and tested on Linux Ubuntu 18.04 LTS. Although not validated nor tested on other platforms and OSs, application should work as long as pre-requisites are met.

### Prerequisites
- OpenVINO    2019.3.376
- Python      3.6.9
- numpy       1.17.4
- cv2         4.1.2-openvino
- PyAutoGUI   0.9.52
- image       1.5.27
- Pillow      6.2.1
- requests    2.22.0
- virtualenv  16.7.9
- ipdb        0.12.3
- ipython     7.10.2


### System Information
- Intel(R) Core(TM) i7-6600 CPU (Skylake)
- Intel(R) HD Graphics 520 (Skylake)
- Ubuntu 18.04.5 :LTS (64-bit)
- 16.0 GB RAM

### Tree structure
You can find the directory structure and corresponding files below.
```
computer-pointer-app/
├── bin
│   └── demo.mp4
├── demo_computer_pointer_app.sh
├── face_detection.py
├── facial_landmarks_detection.py
├── gaze_estimation.py
├── head_pose_estimation.py
├── input_feeder.py
├── intel
│   ├── face-detection-adas-0001
│   │   ├── FP16
│   │   │   ├── face-detection-adas-0001.bin
│   │   │   └── face-detection-adas-0001.xml
│   │   └── FP32
│   │       ├── face-detection-adas-0001.bin
│   │       └── face-detection-adas-0001.xml
│   ├── gaze-estimation-adas-0002
│   │   ├── FP16
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   └── FP32
│   │       ├── gaze-estimation-adas-0002.bin
│   │       └── gaze-estimation-adas-0002.xml
│   ├── head-pose-estimation-adas-0001
│   │   ├── FP16
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   └── FP32
│   │       ├── head-pose-estimation-adas-0001.bin
│   │       └── head-pose-estimation-adas-0001.xml
│   └── landmarks-regression-retail-0009
│       ├── FP16
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       └── FP32
│           ├── landmarks-regression-retail-0009.bin
│           └── landmarks-regression-retail-0009.xml
├── main.py
├── models.lst
├── mouse_controller.py
├── output
│   └── benchmark
│       ├── CPU_FP16_async.txt
│       ├── CPU_FP16_sync.txt
│       ├── CPU_FP32_async.txt
│       ├── CPU_FP32_sync.txt
│       ├── GPU_FP16_async.txt
│       ├── GPU_FP16_sync.txt
│       ├── GPU_FP32_async.txt
│       └── GPU_FP32_sync.txt
├── README.md
└── requirements.txt
```

### Setup
This setup instructions assume you have OpenVINO toolkit already installed and configured on your machine. If not please refer to OpenVINO official documentation to install and configure the toolkit.

```
# Create python virtual environment
python3 -m venv computer-pointer-app-venv
# To ctivate python virtual environment
source computer-pointer-app-venv/bin/activate
# To deactivate python virtual environment
deactivate
# Clone computer-pointer-app
git clone https://github.com/avitial/computer-pointer-app
cd computer-pointer-app
# Setup environment variables for OpenVINO
source /opt/intel/openvino/bin/setupvars.sh
# Install computer-pointer-app pre-requisites
python3 -m pip install -r requirements.txt
# If needed, download model files from Open Model Zoo using model_downloader tool
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --list models.lst
# Install other pre-requisites for mouse controller module
sudo apt-get install python3-tk python3-dev
# Change permissions to run demo script
chmod +x demo_computer_pointer_app.sh
# Run demo script
./demo_computer_pointer_app.sh <path-to-input-file> {FP16,FP32} {CPU,GPU} {async,sync}
```
For example:
`./demo_computer_pointer_app.sh bin/demo.mp4 FP32 CPU async`

## Demo
To run a basic demo of this application, you can try the following:
`./demo_computer_pointer_app.sh <path-to-input-file> {FP16,FP32} {CPU,GPU} {async,sync}`
For example:
`./demo_computer_pointer_app.sh bin/demo.mp4 FP32 CPU async`

To use other features and parameter values refer to documentation or simply run the following:
```
python3 main.py \
    -m_fd intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml \
    -m_fl intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
    -m_hpe intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml \
    -m_ge intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml \
    -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so \
    -i ../bin/demo.mp4 \
    -d CPU
```

## Documentation
Command line arguments supported by application, to get this information simply use -h,--help flag.
```
usage: main.py [-h] -m_fd MODEL_FD -m_fl MODEL_FL -m_hpe MODEL_HPE -m_ge
               MODEL_GE -i INPUT [-l CPU_EXTENSION]
               [-d {CPU,GPU,MYRIAD,HDDL,FPGA}] [-pt PROB_THRESHOLD]
               [-s {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30}]
               [-sb {false,true}] [-sf {false,true}] [-t TIMER]
               [-v {true,false}] [-api {sync,async}]

optional arguments:
  -h, --help            show this help message and exit
  -m_fd MODEL_FD, --model_fd MODEL_FD
                        Required. Path to an xml file with a trained face
                        detection model.
  -m_fl MODEL_FL, --model_fl MODEL_FL
                        Required. Path to an xml file with a trained facial
                        landmarks model.
  -m_hpe MODEL_HPE, --model_hpe MODEL_HPE
                        Required. Path to an xml file with a trained head pose
                        estimation model.
  -m_ge MODEL_GE, --model_ge MODEL_GE
                        Required. Path to an xml file with a trained gaze
                        estimation model.
  -i INPUT, --input INPUT
                        Required. cam/webcam, path to video file.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels impl.
  -d {CPU,GPU,MYRIAD,HDDL,FPGA}, --device {CPU,GPU,MYRIAD,HDDL,FPGA}
                        Specify the target device to infer on. Sample will
                        look for a suitable plugin for device specified (CPU
                        by default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for detections
                        filtering (0.95 by default)
  -s {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30}, --speed {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30}
                        Optional. Move mouse cursor every n-number of frames,
                        speed results in faster I/O feel.(15 by default)
  -sb {false,true}, --show_benchmark {false,true}
                        Optional. Print benchmark data. (False by default)
  -sf {false,true}, --show_frames {false,true}
                        Optional. Enable display frames with OpenCV. (True by
                        default)
  -t TIMER, --timer TIMER
                        Optional. Timer (in seconds) for runtime with camera
                        feed as input. (10 by default)
  -v {true,false}, --visualize {true,false}
                        Optional. Show visualization of outputs of
                        intermediate models. (False by default)
  -api {sync,async}, --api_type {sync,async}
                        Optional. Enable using sync/async API. (async value by
                        default)
```

## Benchmarks
The following benchmark runs were performed using the same machine to keep result comparison equivalent. The table below showcases each of the tests executed with different configurations of api_types, devices and precisions. In addition, same input file was used for a total of 594 frames processed across all runs.

## Tests Performed
| run # | api_type | device | precision |
| ----- | -------- | ------ | --------- |
| 1 | Async | CPU | FP32 |
| 2 | Async | CPU | FP16 |
| 3 | Sync | CPU | FP32 |
| 4 | Sync | CPU | FP16 |
| 5 | Async | GPU | FP32 |
| 6 | Async | GPU | FP16 |
| 7 | Sync | GPU | FP32 |
| 8 | Sync | GPU | FP16 |

## Results
The following benchmark results include times for model loading, inference (sync), input and output processing across CPU and GPU devices. These tests were performed using the same machine to keep result comparison equivalent. For more detailed results please refer to benchmark files provided in output/benchmarks. A total of 594 frames were processed by each test.

| run | model_name | load | inference | input | output |
| ----- | -------- | ------ | --------- | ---- | ------| 
| 1 | face-detection-adas-0001 | 0.20640 | 21.38605 | 1.90775 | 0.02418 |
| - | landmarks-regression-retail-0009 | 0.02845 | 0 | 0.05139 | 0.05134 |
| - | head-pose-estimation-adas-0001 | 0.04210 | 0 | 0.05600 | 0.07264 |
| - | gaze-estimation-adas-0002 | 0.05461 | 1.26204 | 0.00009 | 0.01929 |
| 2 | face-detection-adas-0001 | 0.19509 | 21.90726 | 1.98570 | 0.02225 |
| - | landmarks-regression-retail-0009 | 0.04043 | 0 | 0.05108 | 0.05339 |
| - | head-pose-estimation-adas-0001 | 0.07744 | 0 | 0.05552 | 0.07848 |
| - | gaze-estimation-adas-0002 | 0.07015 | 1.35397 | 0.00010 | 0.02070 |
| 3 | face-detection-adas-0001 | 0.19911 | 19.43413 | 1.65168 | 0.01579 |
| - | landmarks-regression-retail-0009 | 0.02842 | 0.38459 | 0.04698 | 0.01508 |
| - | head-pose-estimation-adas-0001 | 0.04518 | 0.94983 | 0.04920 | 0.02999 |
| - | gaze-estimation-adas-0002 | 0.05367 | 1.22709 |0.00009 | 0.01616 |   
| 4 | face-detection-adas-0001 | 0.19059 | 20.84008 | 1.82064 | 0.02350 |
| - | landmarks-regression-retail-0009 | 0.02957 | 0.39322 | 0.05266 | 0.01700 |
| - | head-pose-estimation-adas-0001 | 0.05832 | 0.97944 | 0.05262 | 0.03160 |
| - | gaze-estimation-adas-0002 | 0.06953 | 1.21084 | 0.00009 | 0.01648 |
| 5 | face-detection-adas-0001 | 13.78354 | 23.81135 | 1.85941 | 0.01391 |
| - | landmarks-regression-retail-0009 | 2.13182 | 0 | 0.04637 | 0.03508 |
| - | head-pose-estimation-adas-0001 | 2.55502 | 0 | 0.05193 |  0.05922 |
| - | gaze-estimation-adas-0002 | 4.20432 | 2.06980 | 0.00010 | 0.01141 |
| 6 | face-detection-adas-0001 | 12.84775 | 13.35935 | 1.79804 | 0.01760 |
| - | landmarks-regression-retail-0009 | 1.94708 | 0 | 0.05464 | 0.04106 |
| - | head-pose-estimation-adas-0001 | 2.65997 | 0 | 0.05825 | 0.06031 |
| - | gaze-estimation-adas-0002 | 4.43970 | 2.28536 | 0.00019 | 0.01160 |
| 7 | face-detection-adas-0001 | 13.92661 | 23.02171 | 1.80403 | 0.01385 |
| - | landmarks-regression-retail-0009 | 2.16507 | 0.98511 | 0.04538 | 0.01095 |
| - | head-pose-estimation-adas-0001 | 2.60678 | 1.57519 | 0.05064 | 0.02999 |
| - |  gaze-estimation-adas-0002 | 4.23325 | 2.03543 | 0.00010 | 0.01125 |   
| 8 | face-detection-adas-0001 | 12.9104 | 12.89596 | 1.80257 | 0.01741 |
| - | landmarks-regression-retail-0009 | 1.95136 | 1.87379 | 0.05478 | 0.01258 |
| - | head-pose-estimation-adas-0001 | 2.62124 | 2.51019 | 0.05684 | 0.01772 |
| - | gaze-estimation-adas-0002 | 4.47449 | 2.20863 | 0.00010 | 0.01152 |

### CPU vs GPU
Comparison between devices for loading time, inference time, and input/output pre-processing times.

#### Model Loading
Loading times are considerably slower on GPU than they are on GPU, it typically takes longer to load models on GPU regardless of precision. 

| run | device | precision | model_name | load_time (ms) |
| --- | ------ | --------- | ---------- | -------------- | 
| 3 | CPU | FP32 | landmarks-regression-retail-0009 | 0.02842 |
| 6 | GPU | FP32 | landmarks-regression-retail-0009 | 1.94708 |

| run | device | precision | model_name | load_time (ms) |
| --- | ------ | --------- | ---------- | -------------- | 
| 1 | CPU | FP16 | face-detection-adas-0001 | 0.20640 |
| 7 | GPU | FP16 | face-detection-adas-0001 | 13.92661 | 

#### Input/output pre-processing
The best times for pre-processing of inputs/outputs is very similar and comparable across CPU/GPU. 

| device  | model_name | input (ms) |
| ------  | ---------- | ---------- | 
| CPU | gaze-estimation-adas-0002 | 0.00009 |
| GPU | gaze-estimation-adas-0002 | 0.0001 |

| device | model_name | output (ms) |
| ------ | ---------- | ----------- | 
| CPU | gaze-estimation-adas-0002 | 0.02999 |
| GPU | gaze-estimation-adas-0002 | 0.01141 |

The worst times for pre-processing of inputs/outputs is very similar and comparable across CPU/GPU.

| device | model_name | input (ms) |
| ------ | ---------- | ---------- | 
| CPU | face-detection-adas-0001 | 1.9857 |
| GPU | face-detection-adas-0001 | 1.85941 |

| device | model_name | output (ms) |
| ------ | ---------- | ----------- | 
| CPU | face-detection-adas-0001 | 0.02225 |
| GPU | face-detection-adas-0001 | 0.01391 |

#### Inference
For the purposes of this project, I could not identify easily if there was a drop in model accuracy from using different precisions, from user application perspective it all felt the same across all models used.

The fastest inference given in the tests, on CPU is about 3x faster than on GPU for some models. For example:

| device | model_name | inference (ms) |
| ------ | ---------- | ----------- | 
| CPU | landmarks-regression-retail-0009 | 0.38459 |
| GPU | landmarks-regression-retail-0009 | 0.98511 |

The slowest inference is comparable among CPU/GPU devices, for example:

| device | model_name | inference (ms) |
| ------ | ---------- | ----------- | 
| CPU | face-detection-adas-0001 | 21.90726 |
| GPU | face-detection-adas-0001 | 23.81135 |

### FP32 vs FP16
From previous theory and concepts learned through the course, one might incline loading times for FP32 would take longer when compared to FP16. But this depends and varies between models and between devices. Based on the following results for Gaze Estimation model it's quite the opposite of what we would expect:

| device | model_name | load (ms) |
| ------ | ---------- | ----------- | 
| CPU,FP32 | gaze-estimation-adas-0002 | 0.05461 |
| CPU,FP16 | gaze-estimation-adas-0002 | 0.07015 |
| GPU,FP32 | gaze-estimation-adas-0002 | 4.20432 |
| GPU,FP16 | gaze-estimation-adas-0002 | 4.4397 |

However for Face Detection model on GPU/CPU, the shorter loading behaviour is as expected:

| device | model_name | load (ms) |
| ------ | ---------- | ----------- | 
| CPU, FP32 | face-detection-adas-0001 | 0.2064 |
| CPU, FP16 | face-detection-adas-0001 | 0.19509 |
| GPU, FP32 | face-detection-adas-0001 | 13.78354 |
| GPU, FP16 | face-detection-adas-0001 | 12.84775 |

#### Available Model Precisions
These models can be downloaded using the Model Downloader tool available in the OpenVINO toolkit.

- face-detection-adas-0001: FP32, FP16 
- landmarks-regression-retail-0009: FP32, FP16
- head-pose-estimation-adas-0001: FP32, FP16
- gaze-estimation-adas-0002: FP32, FP16

To download these models simply use the `models.lst` file provided together with `model_downloader.py` tool under `/opt/intel/openvino/deployment_tools/tools/model_downloader/`, make sure to install the pre-requisites from `requirements.in`:
`python3 -m pip install -r /opt/intel/openvino/deployment_tools/tools/model_downloader/requirements.in`

`python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --list models.lst`

### Sync vs Async
Asynchronous inference makes sense when unrelated tasks can be done in parallel. For example, an asynchronous inference can be started for two models who work on data indepedently and don't depend on each other's outputs. While there are many opportunities for parallelism in this application I have chosen two: a second frame from input can be retrieved while the first one is consumed by the pipeline and also run two models at once. In this project, asynchronous inference is done with the models (Landmarks Regression Detection and Head Pose Estimation) in between first phase and last phase of the pipeline .

The Landmark Regression Detection and Head Pose Estimation phases are independent, but they depend on the output from Face Detection phase. The outputs from Landmark Regression and Head Pose Estimation are consumed by Gaze Estimation phase. Consequently, Landmark Regression Detection and Head Pose Estimation can be executed as asynchronous inference requests.

Asynchronous inference can be done by executing unrelated tasks right after inference request is submitted:
1. Submit asynchronous inference request
2. Perform a separate task
3. When the result from submitted async request is required, call wait() function to wait for results. At this point, processing cannot proceed without the results from the asynchronous inference job. 

The results from asynchronous requests could become available at any time, while other (unrelated) tasks are running. So this time is not captured. For this reason if api_type is set to is async, the async inference times for Landmark Regression Detection and Head Pose Estimation inference requests are not captured (set to 0 ms).

In addition when async mode is chosen, a second frame from input is retrieved while the first one is consumed by the pipeline. 

When taking advantage of parallelism where the application permits and if done correctly, typically asynchronous inference can increase performance if compared to synchronous inference.

#### API Mode Pipeline
In asynchronous inference mode, the pipeline phases can be divided in 4:
1. Facial Detection (phase 1), sync.
2. Landmark Regression Detection (phase 2), async. 
3. Head Pose Estimation (phase 3), async.
4. Gaze Estimation (phase 4), sync.

- For phase 1, other jobs cannot run in parallel as phase 2 and 3 cannot start until this phase completes. Hence sync.
- Inference can run in parallel with Head Pose Estimation (phase 3), hence async.
- Inference can run in parallel with Landmark Detection (phase 2), hence async.
- Other jobs cannot run in parallel. The mouse controller depends on the output, hence sync.

In sync inference mode, the pipeline phases can be divided in 4: 
1. Facial Detection (phase 1), sync.
2. Landmark Regression Detection (phase 2), async. 
3. Head Pose Estimation (phase 3), async.
4. Gaze Estimation (phase 4), sync.

- For models that ran in synchronous inference mode, their respective inference times were captured regardless of the phase.

## Stand Out Suggestions

The mouse controller settings are preset for fast and high-precision movement. A feature to increase/decrease UI responsiveness has been added to the application for better feel, and its with the center value by default for faster UI feel. This setting can be set using `-s, --speed` parameter with values in range of [1-30]. It is set to 15 by default. 

### Features
The following features in application provide user with functionality that can be toggled:
- `-s {0-30}, --speed {1-30}`: move mouse cursor every n-number of frames, speed results in faster I/O feel. (15 by default)
- `-sb {false,true}, --show_benchmark {false,true}`, print benchmark data. (False by default)
- `-sf {false,true}, --show_frames {false,true}`: enable display frames with OpenCV. (True by default)
- `-t TIMER, --timer TIMER`: timer (in seconds) for runtime with camera feed as input. (10 by default)
- `-v {true,false}, --visualize {true,false}`: show visualization of outputs of intermediate models. (False by default)
- `-api {sync,async}, --api_type {sync,async}`: enable using sync/async API. (async value by default)

The application runs only if a face (single) is detected on given input. If a face is/isn't detected, the window title updates to
let user know what is going on. For example:
- If a face is detected: "FACE DETECTED! NOW MOVE THE CURSOR WITH YOUR EYES!"
- If a face is not detected: "A SINGLE FACE IS REQUIRED TO BEGIN..."
- If >1 face is detected: "A SINGLE FACE IS REQUIRED TO BEGIN..."


### Bonus Features
The following keyboard keys with focus on output window have the effect as described:
- `Tab key`: toggles api_type between sync/async modes
- `Space key`: enables/disables visualization of intermediate representation models' outputs.

**Note:** `-sf, --show_frames` must be set to true for bonus features to work.


### Async Inference
Performance metrics gathered to compare between api modes and configurations were:
- Application runtime
- Application render
- OpenCV render

The best performance for async mode was for test run #2 (CPU, FP16, async) with:
- Application runtime, 47.47990 s
- Application render, 12.51 FPS
- OpenCV render, 824.50 FPS

The best performance for sync mode was for test run #3 (CPU, FP32, sync) with:
- Application runtime, 46.01438 s
- Application render, 12.91 FPS
- OpenCV render, 889.00 FPS


### Attributions
The code to visualize intermediate results and draw Head Pose and Gaze estimation come from an answer in [this question](knowledge.udacity.com/questions/171017) given by mentor Shibin M.