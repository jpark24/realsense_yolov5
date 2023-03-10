from matplotlib.pyplot import box
import pyrealsense2 as rs
import cv2
import numpy as np
import time
from threading import Lock
from math import tan,pi
import torch
import os
import datetime as dt
from PIL import ImageFont, ImageDraw, Image


prev_time=0
counter = 0
image_num = 0
model_path = './best.pt'
# prev_timestamp =0
frame_data ={"left" : None,
              "right": None,
              "timestamp_ms" : None
            }


class ObjectDetection:
    def __init__(self, model_path):
        torch.cuda.device(0)
        self.cuda = torch.device("cuda")
        self.input_size=640
        self.model_yolo = torch.hub.load("ultralytics/yolov5","custom", path=model_path)

    def detect(self, x, threshold = 0.5):
        self.model_yolo.conf = threshold

        results=self.model_yolo([x],size=self.input_size)

        return results.xyxy[0].cpu().numpy()



"""
Returns R, T transform from src to dst
"""
def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)

"""
Returns a camera matrix K from librealsense intrinsics
"""
def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                     [            0, intrinsics.fy, intrinsics.ppy],
                     [            0,             0,              1]])

"""
Returns the fisheye distortion from librealsense intrinsics
"""
def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])

def callback(frame):
    global frame_data
    if frame.is_frameset():
        frameset = frame.as_frameset()
        f1 = frameset.get_fisheye_frame(1).as_video_frame()
        f2 = frameset.get_fisheye_frame(2).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        right_data = np.asanyarray(f2.get_data())
        ts = frameset.get_timestamp()
        # frame_mutex.acquire()
        frame_data["left"] = left_data
        frame_data["right"] = right_data
        frame_data["timestamp_ms"] = ts

        
        # cv2.imshow("Left", left_data)
        # cv2.imshow("Right", right_data)
        # cv2.waitKey(1)
        # frame_mutex.release()

OD = ObjectDetection(model_path)
pipe = rs.pipeline()
config = rs.config()
pipe.start(config,callback)



try:
    
    # Set up an OpenCV window to visualize the results
    # WINDOW_TITLE = 'Realsense'
    # cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    # Configure the OpenCV stereo algorithm. See
    # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
    # description of the parameters
    window_size = 5
    min_disp = 0
    # must be divisible by 16
    num_disp = 112 - min_disp
    max_disp = min_disp + num_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                   numDisparities = num_disp,
                                   blockSize = 16,
                                   P1 = 8*3*window_size**2,
                                   P2 = 32*3*window_size**2,
                                   disp12MaxDiff = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 32)

    # Retreive the stream and intrinsic properties for both cameras
    profiles = pipe.get_active_profile()
    streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
               "right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
    intrinsics = {"left"  : streams["left"].get_intrinsics(),
                  "right" : streams["right"].get_intrinsics()}

    # Print information about both cameras
    # print("Left camera:",  intrinsics["left"])
    # print("Right camera:", intrinsics["right"])

    # Translate the intrinsics from librealsense into OpenCV
    K_left  = camera_matrix(intrinsics["left"])
    D_left  = fisheye_distortion(intrinsics["left"])
    K_right = camera_matrix(intrinsics["right"])
    D_right = fisheye_distortion(intrinsics["right"])
    (width, height) = (intrinsics["left"].width, intrinsics["left"].height)
    dim = (width,height)
    # Get the relative extrinsics between the left and right camera
    (R, T) = get_extrinsics(streams["left"], streams["right"])

    stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
    stereo_height_px = 300          # 300x300 pixel stereo output
    stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

    # We set the left rotation to identity and the right rotation
    # the rotation between the cameras
    R_left = np.eye(3)
    R_right = R

    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image
    stereo_width_px = stereo_height_px + max_disp
    stereo_size = (stereo_width_px, stereo_height_px)
    stereo_cx = (stereo_height_px - 1)/2 + max_disp
    stereo_cy = (stereo_height_px - 1)/2

    # Construct the left and right projection matrices, the only difference is
    # that the right projection matrix should have a shift along the x axis of
    # baseline*focal_length
    P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0,               0,         1, 0]])
    P_right = P_left.copy()
    P_right[0][3] = T[0]*stereo_focal_px

    # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
    # since we will crop the disparity later
    Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                  [0, 1,       0, -stereo_cy],
                  [0, 0,       0, stereo_focal_px],
                  [0, 0, -1/T[0], 0]])

    # Create an undistortion map for the left and right camera which applies the
    # rectification and undoes the camera distortion. This only has to be done
    # once
    m1type = cv2.CV_32FC1
    (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
    (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
    undistort_rectify = {"left"  : (lm1, lm2),
                         "right" : (rm1, rm2)}
    while True:
        valid = frame_data["timestamp_ms"] is not None
        
        
        if valid : 
            # counter+=1
            undistorted_left = cv2.remap(frame_data["left"],lm1, lm2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            undistorted_right = cv2.remap(frame_data["right"],rm1, rm2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            undistorted_left_copy = undistorted_left.copy()
            result_from_model_l = OD.detect(undistorted_left_copy)
            boxes_l = result_from_model_l[:,0:4]
            scores_l = result_from_model_l[:, 4]
            labels_l = result_from_model_l[:, 5]
            # print(boxes_l)
            # undistorted_right_copy = undistorted_right.copy()
            # result_from_model_l = OD.detect(undistorted_left_copy)
            # result_from_model_r = OD.detect(undistorted_right)
            

            boxes_l = result_from_model_l[:,0:4]
            scores_l = result_from_model_l[:, 4]
            labels_l = result_from_model_l[:, 5]
            
            # boxes_r = result_from_model_r[:,0:4]
            # scores_r = result_from_model_r[:, 4]
            # labels_r = result_from_model_r[:, 5]
            
            
            
            if boxes_l.shape == (1,4):
                
                b_l = boxes_l.astype(int)
                # print(b_l[0,0])
                cv2.rectangle(undistorted_left_copy, (b_l[0,0],b_l[0,1]),(b_l[0,2],b_l[0,3]), 255, 3, cv2.LINE_AA)
                cv2.imshow("left_image_detection",undistorted_left_copy)
            
            else:
                cv2.imshow("left_image_detection",undistorted_left_copy)
            
            cv2.waitKey(1)
            
            #     print("b_l[0]",b_l[0])
            
            # b_r = boxes_r.astype(int)
            # l_r = labels_r

            # print(boxes_l)
            # print(result_from_model_r)
            
            
            # cv2.rectangle(undistorted_right_copy, (b_r[0],b_r[1]),(b_r[2],b_r[3]), 255, 3, cv2.LINE_AA)
            # cv2.imshow("right_image_detection",undistorted_right_copy)
            # cv2.waitKey(1)
            # if counter == 100:
                
            #     cv2.imwrite(f"left_{image_num}.jpg",undistorted_left)
            #     cv2.imwrite(f"right_{image_num}.jpg",undistorted_right)
            #     image_num+=1
            #     counter=0

            
            # cv2.imshow("undistorted_left", undistorted_left)
            # cv2.imshow("undistorted_right", undistorted_right)
            # cv2.waitKey(1)
            
finally:
    pipe.stop()