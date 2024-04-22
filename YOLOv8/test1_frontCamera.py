import time
import cv2
import numpy as np
# import torch
from ultralytics import YOLO
from pepper_connection import pepper_connection  # 确保你有这个模块来管理Pepper的连接

# 加载YOLOv8模型
model = YOLO("/home/hello/pepper/YOLOv8/yolov8_segment_best.pt")

def process_frame(frame):
    """
    将帧转换为YOLOv8模型所需的格式并进行目标检测。
    """
    results = model(frame)
    # 渲染结果
    frame_detected = results[0].plot()
    return frame_detected

def get_camera_frame(session, name, camera_id, resolution, color_space, fps):
    """
    从Pepper的摄像头捕获视频帧并进行处理。
    """
    service = session.service("ALVideoDevice")
    subscriber_id = service.subscribeCamera(name, camera_id, resolution, color_space, fps)
    try:
        while True:
            start_time = time.time()
            frame = service.getImageRemote(subscriber_id)
            if frame and frame[6]:
                width, height = frame[0], frame[1]
                array = np.frombuffer(frame[6], dtype=np.uint8)
                image = array.reshape((height, width, 3))
                
                # 处理帧
                image_detected = process_frame(image)
                
                # 显示结果
                cv2.imshow("Detected Frame", image_detected)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                print("FPS: {:.2f}".format(1 / (time.time() - start_time)))
    finally:
        print("Unsubscribing from camera...")
        service.unsubscribe(subscriber_id)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    session = pepper_connection.get_session()
    get_camera_frame(session, "Cam1", 0, 2, 13, 15)  # 修改为正确的参数值
