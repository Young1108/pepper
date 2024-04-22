import time
import cv2
import numpy as np
from pepper_connection import pepper_connection  # 确保你有这个模块来管理Pepper的连接
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO("/home/hello/pepper/YOLOv8/yolov8_segment_best.pt")

def track_target(tracker_service, center_x, center_y, frame_width, frame_height):
    """
    使用ALTracker服务根据目标在帧中的位置，调整Pepper头部。
    """
    # 估算目标的位置（这里需要根据实际场景进行调整）
    # 假设物体位于Pepper面前约1米处的地面上
    target_distance = 1.0  # 假设的目标距离
    target_height = 0.0    # 假设的目标高度

    # 计算目标相对于Pepper的大致方向
    dx = (center_x - frame_width / 2) / (frame_width / 2)
    dy = (center_y - frame_height / 2) / (frame_height / 2)

    # 根据dx, dy调整目标的位置，这里仅作为示例，需要根据实际情况调整
    target_position = [target_distance, dx * target_distance, target_height]

    # 使用ALTracker追踪目标
    tracker_service.track("bottle", target_position)  # 注意：实际使用时可能需要选择不同的目标类型

def process_frame(frame, tracker_service):
    """
    使用YOLOv8模型处理给定帧，并进行目标检测。
    """
    results = model(frame)

    # 处理检测结果
    detections = results[0].boxes.data  # 获取检测结果
    names = results[0].names  # 获取类别名称

    # 对于检测到的每个对象
    for detection in detections:
        label_id = int(detection[5].item())  # 获取类别ID
        label_name = names[label_id]  # 获取类别名称
        print("Detected:", label_name)

        if label_name == 'bottle':  # 现在我们使用名称来比较
            # 获取边界框的中心坐标
            x1, y1, x2, y2 = detection[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            track_target(tracker_service, center_x, center_y, frame.shape[1], frame.shape[0])

    # 渲染结果
    frame_detected = results[0].plot()  # 使用render方法绘制检测框和标签

    return frame_detected

def get_2D_camera(name, camera_id, resolution, color_space, fps, session):
    video_service = session.service("ALVideoDevice")
    tracker_service = session.service("ALTracker")
    tracker_service.registerTarget("Face", 0.15)  # 这里假设目标是一个小物体，例如"Face"
    tracker_service.setMode("Head")  # 使用头部追踪

    subscriber_id = video_service.subscribeCamera(name, camera_id, resolution, color_space, fps)
    try:
        while True:
            nao_image = video_service.getImageRemote(subscriber_id)
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
            image_detected = process_frame(image_bgr, tracker_service)

            # 显示检测后的帧
            cv2.imshow("Detected Frame", image_detected)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        video_service.unsubscribe(subscriber_id)
        cv2.destroyAllWindows()
        tracker_service.stopTracker()  # 停止追踪

if __name__ == "__main__":
    session = pepper_connection.get_session()
    name = "python_client3"
    camera_id = 0  # 使用前置摄像头
    resolution = 2
    color_space = 11  # RGB
    fps = 30

    get_2D_camera(name, camera_id, resolution, color_space, fps, session)
