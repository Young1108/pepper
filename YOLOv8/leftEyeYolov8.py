import time
import numpy as np
import cv2
from pepper_connection import pepper_connection
from ultralytics import YOLO

stereo = cv2.StereoSGBM_create(
    minDisparity=-3, # 最小视差值 0  best:-2 48 9 P2:32 * 3 * 9 ** 2
    numDisparities=64,    # max_disp has to be dividable by 16 f. E. HH 192, 256 16  
    blockSize=9,   # 5
    P1=8 * 3 * 9 ** 2,    # 8*number_of_image_channels*SADWindowSize**SADWindowSize
    P2=32 * 3 * 9 ** 2,   # 32*number_of_image_channels*SADWindowSize**SADWindowSize
    disp12MaxDiff=-1, # 左右视差图的最大差异值 1
    # additional parameters
    preFilterCap=63, # 63
    uniquenessRatio=15, # 唯一性比率 10
    speckleWindowSize=100, # 视差图中的区域平滑窗口大小 100
    speckleRange=128 # 64/128
)

# 加载YOLO模型
model = YOLO("YOLOv8/yolov8_segment_best.pt")
# model = YOLO("YOLOv8/yolov8_segment_best")
def track_target(session, center_x, center_y, frame_width, frame_height):
    """
    根据目标在帧中的位置，调整Pepper头部。
    """
    motion_service = session.service("ALMotion")

    # 计算目标位置与帧中心的差异
    dx = (center_x - frame_width / 2) / (frame_width / 2)  # 归一化差值
    dy = (center_y - frame_height / 2) / (frame_height / 2)

    # 调整角度的速度，可以根据需要调整这个值
    speed = 0.2

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

def get_stereo_images(image, width):
    left_image = image[:, :width//2, :]
    right_image = image[:, width//2:, :]
    return left_image, right_image

def process_image_for_yolo(image):    # 对图像进行预处理
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def process_frame(session, frame):
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

def get_depth_map(left_image, right_image):
    # 计算深度图
    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    depth_map = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    # print(depth_map)
    return depth_map

def get_stereo_camera(name, camera_id, resolution, color_space, fps, session):
    service = session.service("ALVideoDevice")
    subscriber_id = service.subscribeCamera(name, camera_id, resolution, color_space, fps)

    try:
        frame_counter = 0
        start_time = time.time()
        while True:
            frame = service.getImageRemote(subscriber_id)
            if frame and frame[6]:
                width, height = frame[0], frame[1]
                array = np.frombuffer(frame[6], dtype=np.uint8)
                image = array.reshape((height, width, 3))

                left_image, right_image = get_stereo_images(image, width)
                
                # 将左图像送入YOLO模型进行目标检测
                left_image_processed = process_image_for_yolo(left_image)
                left_image_detected = process_frame(session, left_image_processed)

                # 将you图像送入YOLO模型进行目标检测
                right_image = process_image_for_yolo(right_image)
                # right_image_detected = process_frame(session, right_image_processed)

                # depth
                depth_map = get_depth_map(left_image, right_image)

                # 可视化深度图
                depth_image = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

                frame_counter += 1
                current_time = time.time()
                if current_time - start_time > 0:
                    fps = frame_counter / (current_time - start_time)
                    cv2.putText(left_image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(right_image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Left Camera", left_image_detected)

                cv2.imshow("Right Camera", right_image)
                cv2.imshow("Depth Map", depth_colormap)

                print("FPS: {:.2f}".format(fps))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        print("Unsubscribing from depth camera...")
        service.unsubscribe(subscriber_id)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    session = pepper_connection.get_session()
    name = "Cam3"
    camera_id = 3
    resolution = 15  # AL::kQQ720px2    15  10 or 15  Image of 640x180px
    color_space = 13  # BGR color space
    fps = 15

    get_stereo_camera(name, camera_id, resolution, color_space, fps, session)