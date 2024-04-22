import time
import cv2
import numpy as np
from pepper_connection import pepper_connection  # 确保你有这个模块来管理Pepper的连接
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO("/home/hello/pepper/YOLOv8/yolov8_segment_best.pt")

def track_target(session, center_x, center_y, frame_width, frame_height):
    """
    根据目标在帧中的位置，调整Pepper头部。
    """
    motion_service = session.service("ALMotion")

    # 计算目标位置与帧中心的差异
    dx = (center_x - frame_width / 2) / (frame_width / 2)  # 归一化差值
    dy = (center_y - frame_height / 2) / (frame_height / 2)

    # 调整角度的速度，可以根据需要调整这个值
    speed = 0.15

    # 调整Pepper的头部
    current_yaw = motion_service.getAngles("HeadYaw", True)[0]
    current_pitch = motion_service.getAngles("HeadPitch", True)[0]
    
    new_yaw = current_yaw - dx * speed
    new_pitch = current_pitch + dy * speed

    # 限制头部运动的范围
    new_yaw = max(min(new_yaw, 1.0), -1.0)
    new_pitch = max(min(new_pitch, 0.5), -0.5)

    motion_service.setAngles("HeadYaw", new_yaw, 0.1)
    motion_service.setAngles("HeadPitch", new_pitch, 0.1)

def process_frame(frame, session):
    """
    使用YOLOv8模型处理给定帧，并进行目标检测。
    """
    results = model(frame)

    # 由于检测结果的格式，可能需要根据你的实际模型输出进行调整
    detections = results[0].boxes.data  # 假设这包含了检测到的所有物体的信息
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
            track_target(session, center_x.item(), center_y.item(), frame.shape[1], frame.shape[0])

    # 渲染结果
    frame_detected = results[0].plot()

    return frame_detected

def get_2D_camera(name, camera_id, resolution, color_space, fps, session):
    video_service = session.service("ALVideoDevice")
    subscriber_id = video_service.subscribeCamera(name, camera_id, resolution, color_space, fps)

    try:
        frame_counter = 0
        start_time = time.time()
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
            image_detected = process_frame(image_bgr, session)

            # 计算并显示FPS
            frame_counter += 1
            current_time = time.time()
            if current_time - start_time > 0:
                fps = frame_counter /  (current_time - start_time)
                cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示检测后的帧
            cv2.imshow("Detected Frame", image_detected)
            
            print("FPS: {:.2f}".format(fps))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Unsubscribing from depth camera...")
        video_service.unsubscribe(subscriber_id)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    session = pepper_connection.get_session()
    name = "python_client3"
    camera_id = 0  # 使用前置摄像头
    resolution = 1
        # AL::k16VGA 4	fps:1 to 15	2560x1920px	Top only can't be displayed
    # AL::k4VGA	 3	fps:1 to 30 1280x960px	Top only  3
    # AL::kVGA	 2	fps:1 to 30	640x480px             10
    # AL::kQVGA	 1	fps:1 to 30	320x240px    
    color_space = 11  # RGB
    fps = 30

    get_2D_camera(name, camera_id, resolution, color_space, fps, session)
