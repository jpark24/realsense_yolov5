import pyrealsense2 as rs
import cv2
import numpy as np
import time

prev_time=0
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.pose)
config.enable_stream(rs.stream.fisheye,1)
config.enable_stream(rs.stream.fisheye,2)
# config.enable_stream(rs.stream.color,848,800,rs.format.y8,30)
# profile=pipe.start(config)
pipeline_wrapper = rs.pipeline_wrapper(pipe)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

device_product_line = str(device.get_info(rs.camera_info.product_line))
print(device_product_line)

profile=pipe.start(config)

try:
    for i in range(0,10000):
        frames= pipe.wait_for_frames()
        current_time = time.time()-prev_time
            
        left = frames.get_fisheye_frame(1)
        left_data = np.asanyarray(left.get_data())

        right = frames.get_fisheye_frame(2)
        right_data = np.asanyarray(right.get_data())
        # image = np.asanyarray(pose)

        # print("Left frame",left_data.shape)
        # print("Right frame",right_data.shape)
        cv2.imshow("Left",left_data)
        cv2.imshow("Right",right_data)
        if cv2.waitKey(1)>0:
            break

        pose = frames.get_pose_frame()

        if pose:
            pose_data = pose.get_pose_data()



finally:
    pipe.stop()

