import os
import sys
import cv2
import time
import math
import numpy as np
import logging as log
from datetime import datetime
from argparse import ArgumentParser

from input_feeder import InputFeeder
from mouse_controller import MouseController
from model import Model


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m_fd", "--model_fd", required=True, type=str,
                        help="Required. Path to an xml file with a trained face detection model. ")
    parser.add_argument("-m_fl", "--model_fl", required=True, type=str,
                        help="Required. Path to an xml file with a trained facial landmarks model. ")
    parser.add_argument("-m_hpe", "--model_hpe", required=True, type=str,
                        help="Required. Path to an xml file with a trained head pose estimation model. ")
    parser.add_argument("-m_ge", "--model_ge", required=True, type=str,
                        help="Required. Path to an xml file with a trained gaze estimation model. ")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Required. cam/webcam, path to video file. ")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default="cam",
                        help="MKLDNN (CPU)-targeted custom layers. "
                             "Absolute path to a shared library with the "
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU", choices=['CPU', 'GPU', 'MYRIAD', 'HDDL', 'FPGA'],
                        help="Specify the target device to infer on. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float, default=0.95,
                        help="Optional. Probability threshold for detections filtering "
                             "(0.95 by default)"),
    parser.add_argument("-s", "--speed", required=False, type=int, choices=range(0, 31), default=30,
                        help="Optional. Move mouse cursor every n-number of frames, speed results in faster I/O feel."
                             "(15 by default)"),
    parser.add_argument("-sb", "--show_benchmark", required=False, type=str, choices=['false', 'true'], default='false',
                        help="Optional. Print benchmark data. "
                             "(False by default)"),
    parser.add_argument("-sf", "--show_frames", required=False, type=str, choices=['false', 'true'], default='true',
                        help="Optional. Enable display frames with OpenCV. "
                             "(True by default)")
    parser.add_argument("-t", "--timer", required=False, type=int, default=10,
                        help="Optional. Timer (in seconds) for runtime with camera feed as input. "
                             "(10 by default)")
    parser.add_argument("-v", "--visualize", required=False, type=str, choices=['true', 'false'], default='false',
                        help="Optional. Show visualization of outputs of intermediate models. "
                             "(False by default)")
    parser.add_argument("-api", "--api_type", required=False, type=str, choices=['sync', 'async'], default='async',
                        help="Optional. Enable using sync/async API. "
                             "(async value by default)")

    return parser


# Build_camera_matrix and draw_axes from https://knowledge.udacity.com/questions/171017
# Credit to Author: Shibin M
def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix


# Build_camera_matrix and draw_axes from https://knowledge.udacity.com/questions/171017
# Credit to Author: Shibin M
def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                   [0, 1, 0],
                   [math.sin(yaw), 0, math.cos(yaw)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])
    # R = np.dot(Rz, Ry, Rx)
    # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    # R = np.dot(Rz, np.dot(Ry, Rx))
    R = Rz @ Ry @ Rx
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(R, xaxis) + o
    yaxis = np.dot(R, yaxis) + o
    zaxis = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame


def scale_dims(shape, x, y):
    width = shape[1]
    height = shape[0]
    x = int(x * width)
    y = int(y * height)

    return x, y


# Scale the landmarks to the whole frame size
def scale_landmarks(landmarks, image_shape, orig, image, visualize):
    color = (0, 0, 255)  # RED
    thickness = cv2.FILLED
    num_lm = len(landmarks)
    orig_x = orig[0]
    orig_y = orig[1]
    scaled_landmarks = []
    for point in range(0, num_lm, 2):
        x, y = scale_dims(image_shape, landmarks[point], landmarks[point + 1])
        x_scaled = orig_x + x
        y_scaled = orig_y + y
        if visualize:
            image = cv2.circle(image, (x_scaled, y_scaled), 2, color, thickness)
        scaled_landmarks.append([x_scaled, y_scaled])

    return scaled_landmarks, image


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    # Initialize class variables
    next_frame = input_type = input_stream = None
    face_detected = show_frames = visualize = False
    elapsed_time = render_total = frame_ct = 0
    timer = args.timer
    speed = args.speed
    app_runtime = time.time()
    ge_outputs = [[0, 0, 0]]
    async_mode = True
    color = (0, 255, 0)  # OpenCV setting
    focal_length = 950.0
    scale = 50
    start_time = datetime.now()
    networks_info: Any = {"fd_inf_time": 0, "fl_inf_time": 0, "hpe_inf_time": 0, "ge_inf_time": 0, "fd_input_time": 0,
                          "fl_input_time": 0, "hpe_input_time": 0, "ge_input_time": 0, "fd_output_time": 0,
                          "fl_output_time": 0, "hpe_output_time": 0, "ge_output_time": 0}

    if args.visualize == 'true':
        visualize = True
    if args.show_frames == 'true':
        show_frames = True

    # Initialize models
    fd_infer_network = Model(model_name=args.model_fd, device=args.device, cpu_extension=args.cpu_extension)
    fl_infer_network = Model(model_name=args.model_fl, device=args.device, cpu_extension=args.cpu_extension)
    hpe_infer_network = Model(model_name=args.model_hpe, device=args.device, cpu_extension=args.cpu_extension)
    ge_infer_network = Model(model_name=args.model_ge, device=args.device, cpu_extension=args.cpu_extension)

    # Load models
    networks_info["fd"] = fd_infer_network.load_model()
    networks_info["fl"] = fl_infer_network.load_model()
    networks_info["hpe"] = hpe_infer_network.load_model()
    networks_info["ge"] = ge_infer_network.load_model()

    # Initialize mouse controller
    mc = MouseController('high', 'fast')
    screen_width, screen_height = mc.monitor()
    mc.put(int(screen_width / 2), int(screen_height / 2))  # Place the mouse cursor in the center of the screen

    # Handle input stream
    try:
        if args.input == 'cam' or args.input == 'webcam' or args.input == '0':  # if single webcam is available
            input_type = 'cam'
            input_stream = InputFeeder(input_type='cam', input_file=2)
        elif args.input.endswith('.mp4'):  # check for valid video format
            if not (os.path.exists(args.input)):
                log.error("Specified input file doesn't exist")
                return 1
            input_type = 'video'
            input_stream = InputFeeder(input_type='video', input_file=args.input)
        else:
            log.error("Specified input file {} doesn't exist".format(args.input))
    except Exception as e:
        log.error("ERROR: ", e)

    input_stream.load_data(0)

    print("To close the application, press 'ESC' or 'q' with focus on the output window")

    ret, frame = input_stream.read_data()
    ret, next_frame = input_stream.read_data()

    if args.api_type != 'async':
        async_mode = False

    cur_request_id = 0
    next_request_id = 1
    # Loop until input stream is over or timer lapses and cam
    while input_stream.is_opened():
        elapsed_time = (datetime.now() - start_time).total_seconds()
        detections = False  # reset detections
        # Read from the video capture
        if async_mode:
            ret, next_frame = input_stream.read_data()  # get opencv frame
            frame_ct += 1
        else:
            ret, frame = input_stream.read_data()
            frame_ct += 1
        if not ret:
            break

        if async_mode:  # async inference
            fd_outputs, fd_inf_time, fd_input_time, fd_output_time = exec_pipeline(fd_infer_network, next_frame,
                                                                                   next_request_id, False , None,
                                                                                   None)  # sync inference
            networks_info["fd_inf_time"] += fd_inf_time
            networks_info["fd_input_time"] += fd_input_time
            networks_info["fd_output_time"] += fd_output_time

            detections = [[label, conf, xmin, ymin, xmax, ymax] for _, label, conf, xmin, ymin, xmax, ymax
                          in fd_outputs[fd_infer_network.output_blob][0][0] if conf >= args.prob_threshold]


            if len(detections) == 1:
                face_detected = True
                for detection in detections:
                    label, conf, xmin, ymin, xmax, ymax = detection
                    width = next_frame.shape[1]
                    height = next_frame.shape[0]
                    xmin = int(xmin * width)
                    ymin = int(ymin * height)
                    xmax = int(xmax * width)
                    ymax = int(ymax * height)
                    face_frame = next_frame[ymin:ymax, xmin:xmax]

                    fl_outputs, fl_inf_time, fl_input_time, fl_output_time = exec_pipeline(
                        fl_infer_network, face_frame,
                        cur_request_id, async_mode,
                        None, None)  # async inference
                    networks_info["fl_input_time"] += fl_input_time
                    networks_info["fl_output_time"] += fl_output_time

                    scaled_lm, next_frame = scale_landmarks(landmarks=fl_outputs[fl_infer_network.output_blob][0], image_shape=face_frame.shape,
                                                       orig=(xmin, ymin), image=next_frame, visualize=visualize)

                    hpe_outputs, hpe_inf_time, hpe_input_time, hpe_output_time = exec_pipeline(
                        hpe_infer_network, face_frame,
                        cur_request_id, async_mode,
                        None, None)  # async inference
                    networks_info["hpe_input_time"] += hpe_input_time
                    networks_info["hpe_output_time"] += hpe_output_time
                    hp_angles = [hpe_outputs['angle_y_fc'][0], hpe_outputs['angle_p_fc'][0],
                                 hpe_outputs['angle_r_fc'][0]]
                    ge_outputs, ge_inf_time, ge_input_time, ge_output_time = exec_pipeline(ge_infer_network, next_frame,
                                                                                           cur_request_id, False,
                                                                                           scaled_lm, [hp_angles])  # sync inference
                    networks_info["ge_inf_time"] += ge_inf_time
                    networks_info["ge_input_time"] += ge_input_time
                    networks_info["ge_output_time"] += ge_output_time

                    if speed == 0:  # move mouse once every speed
                        mc.move(ge_outputs[ge_infer_network.output_blob][0][0], ge_outputs[ge_infer_network.output_blob][0][1])
                        speed = args.speed
                    speed = speed - 1

            else:
                face_detected = False

        else:  # sync inference
            fd_outputs, fd_inf_time, fd_input_time, fd_output_time = exec_pipeline(fd_infer_network, frame,
                                                                                   cur_request_id, async_mode, None,
                                                                                   None)
            networks_info["fd_inf_time"] += fd_inf_time
            networks_info["fd_input_time"] += fd_input_time
            networks_info["fd_output_time"] += fd_output_time

            detections = [[label, conf, xmin, ymin, xmax, ymax] for _, label, conf, xmin, ymin, xmax, ymax
                          in fd_outputs[fd_infer_network.output_blob][0][0] if conf >= args.prob_threshold]

            if len(detections) == 1:
                face_detected = True
                for detection in detections:
                    label, conf, xmin, ymin, xmax, ymax = detection
                    width = frame.shape[1]
                    height = frame.shape[0]
                    xmin = int(xmin * width)
                    ymin = int(ymin * height)
                    xmax = int(xmax * width)
                    ymax = int(ymax * height)
                    face_frame = frame[ymin:ymax, xmin:xmax]

                    fl_outputs, fl_inf_time, fl_input_time, fl_output_time = exec_pipeline(fl_infer_network, face_frame,
                                                                                           cur_request_id, async_mode,
                                                                                           None, None)
                    networks_info["fl_inf_time"] += fl_inf_time
                    networks_info["fl_input_time"] += fl_input_time
                    networks_info["fl_output_time"] += fl_output_time

                    scaled_lm, frame = scale_landmarks(landmarks=fl_outputs[fl_infer_network.output_blob][0],
                                                       image_shape=face_frame.shape,
                                                       orig=(xmin, ymin), image=frame, visualize=visualize)

                    hpe_outputs, hpe_inf_time, hpe_input_time, hpe_output_time = exec_pipeline(hpe_infer_network,
                                                                                               face_frame,
                                                                                               cur_request_id,
                                                                                               async_mode, None,
                                                                                               None)
                    networks_info["hpe_inf_time"] += hpe_inf_time
                    networks_info["hpe_input_time"] += hpe_input_time
                    networks_info["hpe_output_time"] += hpe_output_time

                    hp_angles = [hpe_outputs['angle_y_fc'][0], hpe_outputs['angle_p_fc'][0],
                                 hpe_outputs['angle_r_fc'][0]]

                    ge_outputs, ge_inf_time, ge_input_time, ge_output_time = exec_pipeline(ge_infer_network, frame,
                                                                                           cur_request_id, async_mode,
                                                                                           scaled_lm, [hp_angles])
                    networks_info["ge_inf_time"] += ge_inf_time
                    networks_info["ge_input_time"] += ge_input_time
                    networks_info["ge_output_time"] += ge_output_time

                    if speed == 0:  # move mouse once every speed
                        mc.move(ge_outputs[ge_infer_network.output_blob][0][0], ge_outputs[ge_infer_network.output_blob][0][1])
                        speed = args.speed
                    speed = speed - 1
            else:
                face_detected = False

        if visualize and show_frames is True:
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            center_face = (xmin + face_frame.shape[1] / 2, ymin + face_frame.shape[0] / 2, 0)
            frame = draw_axes(frame, center_face, hp_angles[0], hp_angles[1], hp_angles[2], scale, focal_length)
            frame = draw_axes(frame, scaled_lm[0], ge_outputs[ge_infer_network.output_blob][0][0], ge_outputs[ge_infer_network.output_blob][0][1], ge_outputs[ge_infer_network.output_blob][0][2], scale,
                              focal_length)
            frame = draw_axes(frame, scaled_lm[1], ge_outputs[ge_infer_network.output_blob][0][0], ge_outputs[ge_infer_network.output_blob][0][1], ge_outputs[ge_infer_network.output_blob][0][2], scale,
                              focal_length)

        # Process frame(s) if input is directory or single image
        # Logic to show frame(s) in OpenCV
        render_start = time.perf_counter()
        if show_frames is True:
            input_stream.show_frame(frame, face_detected)
        render_total += time.perf_counter() - render_start

        key = cv2.waitKey(1)
        if key in {27, 113, 227}:  # q, esc, ctrl+c
            print("Close key pressed, exiting.")
            break
        if key == 9:  # tab
            async_mode = not async_mode
            print("Switched to {} mode".format("async" if async_mode else "sync"))
        if key == 32:
            visualize = not visualize
            print("Turned {} model visualization".format("on" if visualize else "off"))
        # Stop app after timer lapsed and input type is cam
        if elapsed_time > timer and input_type == 'cam':
            print("Webcam feed timer of {}s lapsed, exiting.".format(args.timer))
            break

        if async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

    # Close stream
    input_stream.close()
    cv2.destroyAllWindows()
    # Display benchmark results

    if args.show_benchmark == 'true':
        app_runtime = time.time() - app_runtime
        show_benchmark(args, networks_info, app_runtime, frame_ct, render_total)


def exec_pipeline(network, input_image, request_id, async_mode, landmarks, head_pose_angles):
    input_time = time.perf_counter()
    p_image = input_image
    input_time = time.perf_counter() - input_time
    output_time = 0
    det_time = 0

    if landmarks is None:
        input_time = time.perf_counter()
        p_image = network.preprocess_input(input_image)
        input_time = time.perf_counter() - input_time

    if async_mode is True:  # async inference time not required
        res = network.predict(p_image, request_id, async_mode, landmarks, head_pose_angles)
    else:  # calculate inference time
        inf_start = time.perf_counter()
        res = network.predict(p_image, request_id, async_mode, landmarks, head_pose_angles)
        det_time = time.perf_counter() - inf_start

    while network.wait(request_id) != 0 and async_mode is True:  # wait for async to complete
        pass

    if network.wait(request_id) == 0 and async_mode is True:  # verify if async request completed
        output_time = time.perf_counter()
        output = network.preprocess_output(res, request_id)
        output_time = time.perf_counter() - output_time
    else:  # sync inference
        output_time = time.perf_counter()
        output = network.preprocess_output(res, 0)
        output_time = time.perf_counter() - output_time

    return output, det_time, input_time, output_time


def show_benchmark(args, networks_info, app_runtime, frame_count, render_total):
    total_inf_time = networks_info['fd_inf_time'] + networks_info['fl_inf_time'] + networks_info['hpe_inf_time'] + \
                     networks_info['ge_inf_time']
    model_load_time = networks_info['fd'][1] + networks_info['fl'][1] + networks_info['hpe'][1] + networks_info['ge'][1]
    app_fps = frame_count / app_runtime
    print('\n*****************************************')
    print('*********** BENCHMARK RESULTS ***********')
    print('*****************************************')
    print(' Device: {}\n API type: {}\n Model Metrics:'.format(args.device, args.api_type))
    print('  {}\n\tload time:\t   {:.5f} ms\n\tsync inference:\t   {:.5f} ms'.format(networks_info['fd'][0],
                                                                                     networks_info['fd'][1],
                                                                                     networks_info['fd_inf_time']))
    print('   \tinput processing:  {:.5f} ms\n\toutput processing: {:.5f} ms'.format(networks_info['fd_input_time'],
                                                                                     networks_info['fd_output_time']))
    print('\tbatch size: {}, precision: {}'.format(networks_info['fd'][2], networks_info['fd'][3]))
    print('  {}\n\tload time:\t   {:.5f} ms\n\tsync inference:\t   {:.5f} ms'.format(networks_info['fl'][0],
                                                                                     networks_info['fl'][1],
                                                                                     networks_info['fl_inf_time']))
    print('   \tinput processing:  {:.5f} ms\n\toutput processing: {:.5f} ms'.format(networks_info['fl_input_time'],
                                                                                     networks_info['fl_output_time']))
    print('\tbatch size: {}, precision: {}'.format(networks_info['fl'][2], networks_info['fl'][3]))
    print('  {}\n\tload time:\t   {:.5f} ms\n\tsync inference:\t   {:.5f} ms'.format(networks_info['hpe'][0],
                                                                                     networks_info['hpe'][1],
                                                                                     networks_info['hpe_inf_time']))
    print('   \tinput processing:  {:.5f} ms\n\toutput processing: {:.5f} ms'.format(networks_info['hpe_input_time'],
                                                                                     networks_info['hpe_output_time']))
    print('\tbatch size: {}, precision: {}'.format(networks_info['hpe'][2], networks_info['hpe'][3]))
    print('  {}\n\tload time:\t   {:.5f} ms\n\tsync inference\t   {:.5f} ms'.format(networks_info['ge'][0],
                                                                                    networks_info['ge'][1],
                                                                                    networks_info['ge_inf_time']))
    print('   \tinput processing:  {:.5f} ms\n\toutput processing: {:.5f} ms'.format(networks_info['ge_input_time'],
                                                                                     networks_info['ge_output_time']))
    print('\tbatch size: {}, precision: {}'.format(networks_info['ge'][2], networks_info['ge'][3]))
    print(' Models loading time:\t   {:.5f} ms'.format(model_load_time))
    print(' Models sync inference:  {:.5f} ms'.format(total_inf_time))
    print(' App runtime:\t\t  {:.5f}  s'.format(app_runtime))
    print(' Render time:\t\t   {:.5f} ms '.format(render_total))
    print(' Processed frames: {}'.format(frame_count))
    if render_total != 0:
        render_fps = frame_count/render_total
    else: render_fps = 0.0
    print(' App:\t\t   {:.2f} FPS\n OpenCV Render:   {:.2f} FPS '.format(app_fps, render_fps))
    print('******************************************\n')


def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('CTRL+C pressed, interrupting app.')
        try:
            sys.exit(0)
        except SystemExit:
            os.exit(0)
