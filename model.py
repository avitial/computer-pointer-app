'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import sys
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Model:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device, cpu_extension):
        self.plugin = device
        self.model = model_name
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.device = device
        self.cpu_extension = cpu_extension
        self.left_eye_input = 'left_eye_image'
        self.right_eye_input = 'right_eye_image'
        self.head_pose_input = 'head_pose_angles'

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # Load the model
        self.model_xml = self.model
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        log.info("Loading {} model...".format(self.model_xml))
        self.ie = IECore()

        # Read IR as IENetwork
        try:
            self.network = IENetwork(model=self.model_xml, weights=self.model_bin)
            log.info("Network initialized")

        except Exception as e:
            log.error("Could not initialize network. Check if model file/path is valid. {}:", e)

        self.input_blob = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.network.outputs))

        start_time = time.perf_counter()
        self.check_model()  # check for supported layers

        # Return the loaded inference plugin
        log.info("Reading IR...")
        self.exec_network = self.ie.load_network(network=self.network, num_requests=2, device_name=self.device)
        load_time = time.perf_counter() - start_time
        batch_size = self.network.batch_size
        precision = self.network.precision
        model_name = self.model_xml.rsplit("/", 1)[1]
        log.info('Network batch size: {}, precision: {}'.format(self.network.batch_size, self.network.precision))

        return model_name, load_time, batch_size, precision

    def predict(self, image, request_id, async_mode, landmarks=None, head_pose_angles=None):
         '''
         You will need to complete this method.
         This method is meant for running predictions on the input image.
         '''
         if landmarks is not None:
              self.input_shape = self.network.inputs[self.left_eye_input].shape
              eye_offset = int(self.input_shape[2] / 2)
              right_eye = landmarks[0]
              left_eye = landmarks[1]
              left_crop = image[left_eye[1] - eye_offset:left_eye[1] + eye_offset,
                          left_eye[0] - eye_offset:left_eye[0] + eye_offset]
              right_crop = image[right_eye[1] - eye_offset:right_eye[1] + eye_offset,
                          right_eye[0] - eye_offset: right_eye[0] + eye_offset]
              eye_images = list(map(self.preprocess_input, [left_crop, right_crop]))
              input_blob = {self.left_eye_input: eye_images[0], self.right_eye_input: eye_images[1],
                          self.head_pose_input: head_pose_angles}

              if async_mode is True:  # async
                  return self.exec_network.start_async(request_id=request_id,
                                                       inputs=input_blob)
              else:  # sync
                  return self.exec_network.infer(inputs=input_blob)

         else:
              if async_mode is True:
                   return self.exec_network.start_async(request_id=request_id, inputs={self.input_blob: image})  # async
              else:
                   return self.exec_network.infer(inputs={self.input_blob: image})  # sync


    def wait(self, request_id):
        return self.exec_network.requests[request_id].wait(-1)

    def check_model(self):
        log.info("Create Inference Engine...")
        self.ie = IECore()

        # Add any necessary extensions
        if self.device == 'CPU' and self.cpu_extension:
            self.ie.add_extension(self.cpu_extension, self.device)
            log.info("Loaded extensions to {}".format(self.device))

        # Check for unsupported layers, continue if all layers supported
        supported_layers = self.ie.query_network(self.network, self.device)
        not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0 and self.device == 'CPU':
            log.error("These layers are not supported by the plugin for device {}:\n {}"
                      .format(self.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        n, c, h, w = self.input_shape

        new_image = cv2.resize(image, (w, h))
        new_image = new_image.transpose((2, 0, 1))
        new_image = new_image.reshape((n, c, h, w))

        return new_image

    def preprocess_output(self, output, request_id):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # return self.exec_network.requests[request_id].outputs[self.output_blob]
        return self.exec_network.requests[request_id].outputs
