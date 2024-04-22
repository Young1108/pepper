import time
import cv2
import numpy as np
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

def get_2D_camera(name, camera_id, resolution, color_space, fps, session):
    """
    从Pepper的2D摄像头捕获视频帧，使用YOLOv8模型进行目标检测，并显示处理后的视频帧。
    """
    video_service = session.service("ALVideoDevice")
    video_client = video_service.subscribeCamera(name, camera_id, resolution, color_space, fps)

    try:
        while True:
            start_time = time.time()
            nao_image = video_service.getImageRemote(video_client)

            if nao_image is None:
                print("Failed to get image")
                continue

            # 将获取的图像转换为OpenCV格式
            image_width = nao_image[0]
            image_height = nao_image[1]
            array = np.frombuffer(nao_image[6], dtype=np.uint8)
            image = array.reshape((image_height, image_width, 3))

            # YOLO模型需要BGR格式的图像
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 处理帧并进行目标检测
            image_detected = process_frame(image_bgr)

            # 显示检测后的帧
            cv2.imshow("Detected Frame", image_detected)

            # 计算并显示FPS
            fps = 1 / (time.time() - start_time)
            print("FPS: {:.2f}".format(fps))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Unsubscribing from camera...")
        video_service.unsubscribe(video_client)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    session = pepper_connection.get_session()
    name = "python_client"
    camera_id = 0  # 使用前置摄像头
    resolution = 2  # VGA
    color_space = 11  # RGB
    fps = 30

    get_2D_camera(name, camera_id, resolution, color_space, fps, session)