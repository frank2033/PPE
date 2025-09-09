import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import torch  

# 1. 仅执行一次：导出 TensorRT 引擎（成功后可注释此段）
# ！！！首次运行时执行，生成 engine 文件后，后续推理可跳过此步！！！
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False  # 适配 Jetson 调度逻辑
model_pt = YOLO("/home/nvidia/Desktop/ppe/model/yolo11s_ppe_best.pt")
# 导出 TensorRT 引擎（指定 device=0，强制用 GPU 生成）
model_pt.export(
    format="engine",
    device=0,
    imgsz=640,  # 与相机分辨率一致（640x480，取 640 确保适配）
    half=True  # 启用 FP16 精度，减少显存占用，提升推理速度
)
print("TensorRT 引擎导出完成！文件路径：yolo11s_ppe_best.engine")


# 2. 配置 D435i 相机（保持原逻辑，增加稳定性判断）
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

def get_aligned_images():
    frames = pipeline.wait_for_frames()
    if not frames:
        return None, None
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None
    depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
    color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
    return depth_image, color_image


# 3. 加载 TensorRT 引擎推理（性能最优，避免 cuDNN 调用）
# ！！！引擎导出成功后，后续运行可直接加载此 engine 文件！！！
model_engine = YOLO("/home/nvidia/Desktop/ppe/model/yolo11s_ppe_best.engine")  # 替换为实际 engine 路径
model_engine.to("cuda:0")  # 强制加载到 GPU


# 4. 窗口配置与推理循环
cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RealSense", 640, 480)

try:
    while True:
        img_depth, img_color = get_aligned_images()
        if img_color is None:
            continue

        # 推理：用 TensorRT 引擎，禁用 CPU 回退，确保 GPU 执行
        results = model_engine.predict(
            img_color,
            conf=0.5,
            stream=False,
            device="cuda:0",  # 固定 GPU 设备
            batch=1,
            verbose=False,
            half=True  # 与导出引擎时的精度一致
        )

        # 处理结果（增加空判断，避免报错）
        annotated = img_color.copy()
        if results and len(results) > 0:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                annotated = r.plot()

        cv2.imshow("RealSense", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    print("资源已正常释放")
