#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import numpy as np
import cv2
import time
import sys
import logging
import datetime
from logging.handlers import RotatingFileHandler
import sys
import os
import errno
import argparse
import warnings
warnings.filterwarnings("ignore")
class ObjectDetection:

    def __init__(self):

        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--inputType", type=int, default=1,
                    help="# 0 for detection on img, 1 for detection on video, 2 for detection through webcam(#will work only when docker run in virtualbox with window/mac and on linux use --device=/dev/video0:/dev/video0)")
        ap.add_argument("-i", "--input", type=str, default = "videoplayback.mp4",
                    help="complete name of input file with .ext, don't use when -type = 2")
        ap.add_argument("-o", "--output", type=str, default="",
                    help="complete name of output file with .ext, #incase of video add .mp4 and image add .png ")

        self.args = vars(ap.parse_args())

        log_folder_path = os.path.join(os.getcwd(),"log")
        try:
            os.mkdir(log_folder_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
        else:
            print("Log Directory Created")
        #Creating File Logger for debugging    
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(thread)d] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_hdlr = RotatingFileHandler('log/debug_{}'.format(datetime.datetime.now().strftime('%Y_%m_%d')), maxBytes= 10*1024*1024, backupCount = 10)
        file_hdlr.setFormatter(formatter)
        file_logger = logging.getLogger('file_logger')
        file_logger.setLevel(logging.INFO)
        file_logger.addHandler(file_hdlr)
        #Creating File logger for total person detected
        formatter_app = logging.Formatter('%(asctime)s.%(msecs)03d [%(thread)d] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_hdlr_app = RotatingFileHandler('log/App_Info_{}'.format(datetime.datetime.now().strftime('%Y_%m_%d')), maxBytes= 10*1024*1024, backupCount = 10)
        file_hdlr_app.setFormatter(formatter_app)
        file_logger_app = logging.getLogger('file_logger_app')
        file_logger_app.setLevel(logging.INFO)
        file_logger_app.addHandler(file_hdlr_app)
        #creating stream handle for console display
        sys_hdlr = logging.StreamHandler(sys.stdout)
        sys_hdlr.setFormatter(formatter)
        sys_logger = logging.getLogger('sys_logger')
        sys_logger.setLevel(logging.INFO)
        sys_logger.addHandler(sys_hdlr)
        
        
        self.model = self.load_model()
        self.model.conf = 0.5
        self.model.classes = [0] 
        self.device = 'cpu'
        
    def log_message(self, message, std_out = None):
        file_logger = logging.getLogger('file_logger')
        sys_logger = logging.getLogger('sys_logger')
        file_logger_app = logging.getLogger('file_logger_app')
        if std_out != 3:
            file_logger.info(message)
            if std_out is not None:
                sys_logger.info(message)
            elif std_out == 2:
                file_logger.exception(message)
        elif std_out == 3:
            file_logger_app.info(message)
            

    def load_model(self):
        """
        This function will load pretrained yolov5 model.
        """
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        except Exception as e:
            self.log_message("Error Loading Model")
            self.log_message(e,2)
            sys.exit('OOPS!! Something Went Wrong.')
        else:
            self.log_message("Model Loaded Successfully")    
        return model
    
    def read_video(self, input_file):
        """
        Function reads the video from the file frame by frame.
        :param input_file: input file path
        :return:  OpenCV object to stream video frame by frame.
        """
        cap = cv2.VideoCapture(input_file)
        assert cap is not None
        return cap

    def get_score(self, frame):
        """
        function performs inference on each frame.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.
        """
        try:
            self.model.to(self.device)
            results = self.model([frame])
            labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
        except Exception as e:
            self.log_message("Error generating Inference")
            self.log_message(e,2)
            sys.exit('OOPS!! Something Went Wrong.')
        return labels, cord

    def plot_bb(self, results, frame):
        """
        plots bounding box and labels on frame.
        :param results: inferences made by model
        :param frame: frame on which to  make the bounding box
        :return: new frame with boxes and labels plotted.
        """
        try:
            labels, cord = results
            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            for i in range(n):
                row = cord[i]
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                label = f"{int(row[4]*100)}"
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, f"Total Person Detected: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #print(f"Total Targets: {n}")
                self.log_message(f"Total Person Detected: {n}",3)
        except Exception as e:
            self.log_message("Error Plotting Boxes")
            self.log_message(e,2)
            sys.exit('OOPS!! Something Went Wrong.')

        return frame

    def __call__(self):
        self.log_message("Parameter list:{}".format(self.args))
        if self.args["inputType"] == 0:
            try:
                frame = cv2.imread(os.path.join("input",self.args['input']))
            except Exception as e:
                self.log_message("Error Reading File")
                self.log_message(e,2)
                sys.exit('OOPS!! Something Went Wrong.')
            results = self.get_score(frame)
            frame = self.plot_bb(results,frame)
            if self.args['output'] != "":
                out_file = os.path.join("output",self.args['output'])
            else:
                out_file = os.path.join("output","Labeled_IMG.png")
            try:
                cv2.imwrite(out_file,frame)
            except Exception as e:
                self.log_message("Error writing file")
                self.log_message(e,2)
                sys.exit('OOPS!! Something Went Wrong.')
            else:
                self.log_message("Output Generated Successfully")

        elif self.args["inputType"] == 1:
            player = self.read_video(os.path.join("input",self.args['input'])) # create streaming service for application
            assert player.isOpened()
            x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
            y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
            four_cc = cv2.VideoWriter_fourcc(*"MJPG")
            if self.args['output'] != "":
                out_file = os.path.join("output",self.args['output'])
            else:
                out_file = os.path.join("output","Labeled_Video.mp4")
            out = cv2.VideoWriter(out_file, four_cc, 20, (x_shape, y_shape))

            while True:
                ret, frame = player.read()
                if not ret:
                    break
                results = self.get_score(frame)
                frame = self.plot_bb(results, frame)
                out.write(frame)
            self.log_message("Output Generated Successfully")
            player.release()

        elif self.args["inputType"] == 2:
        
            print("[INFO] starting video stream...")
            try:
                vs = cv2.VideoCapture(0)  #will work only when docker run in virtualbox with window/mac and on linux use --device=/dev/video0:/dev/video0.
                assert vs.isOpened()
                x_shape = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
                y_shape = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
                four_cc = cv2.VideoWriter_fourcc(*"MJPG")
                if self.args['output'] != "":
                    out_file = os.path.join("output",self.args['output'])
                else:
                    out_file = os.path.join("output","Labeled_Video_RealTime.mp4")
                out = cv2.VideoWriter(out_file, four_cc, 20, (x_shape, y_shape))
            except Exception as e:
                self.log_message("Error Getting Camera")
                self.log_message(e,2)
            time.sleep(2.0)
            while True:
                try:
                    ret, frame = vs.read()
                except Exception as e:
                    self.log_message("Error Reading Frames")
                    self.log_message(e,2)
                    sys.exit('OOPS!! Something Went Wrong.')
                if not ret:
                    break
                results = self.get_score(frame)
                frame = self.plot_bb(results, frame)
                out.write(frame)
                cv2.imshow("Detected Object- Press e to exit", frame)
                if cv2.waitKey(5) & 0xFF == ord('e'):
                    break
            cv2.destroyAllWindows()    
            vs.release()



# In[14]:


a = ObjectDetection()
a()


# In[ ]:




