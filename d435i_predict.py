import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import torch          
import os

# 解决CuDNN错误：禁用CuDNN加速（如果兼容问题无法解决）
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.enabled = False  # 禁用CuDNN

# 1. 加载模型
model = YOLO("/home/nvidia/Desktop/ppe/model/yolo11s_ppe_best.pt")

# 2. 配置并启动相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image

# 解决Qt线程错误：使用cv2的WINDOW_NORMAL替代AUTOSIZE
cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)

try:
    while True:
        img_depth, img_color = get_aligned_images()
        if img_color is None:
            continue

        # 调整预测参数，避免流处理可能的线程问题
        results = model.predict(img_color, conf=0.5, stream=True, device="cuda:0")  # 移除stream=True
        for r in results:
            if torch.isnan(r.boxes.xyxy).any():
                annotated = img_color
            else:
                annotated = r.plot()
            break

        cv2.imshow("RealSense", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
finally:
    cv2.destroyAllWindows()
    pipeline.stop()
