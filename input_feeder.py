'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type = input_type
        if input_type == 'video' or input_type == 'image' or input_type == 'dir':
            self.input_file = input_file
    
    def load_data(self, index):
        if self.input_type == 'video':
            self.cap = cv2.VideoCapture(self.input_file)
        elif self.input_type == 'cam':
            self.cap = cv2.VideoCapture(0)

    def read_data(self):
        ret, frame = self.cap.read()
        if self.input_type == 'cam':
            frame = cv2.flip(frame, 1)  # flip horizontal to mirror cam feed face movement
        return ret, frame

    def set_property(self):
        self.cap.set(property=cv2.CAP_PROP_POS_FRAMES, val=0)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            for _ in range(10):
                _, frame = self.cap.read()
            yield frame

    def is_opened(self):
        while self.cap.isOpened():
            return True

    def show_frame(self, frame, face_detected):
        msg1 = "FACE DETECTED! NOW MOVE THE CURSOR WITH YOUR EYES!"
        msg2 = "A SINGLE FACE IS REQUIRED TO BEGIN..."


        self.frame = cv2.imshow('', frame)

        if face_detected:
            self.frame = cv2.setWindowTitle('', msg1)
        else:
            self.frame = cv2.setWindowTitle('', msg2)

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type == 'image':
            self.cap.release()
