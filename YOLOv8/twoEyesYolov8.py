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

def get_stereo_images(image, width):
    left_image = image[:, :width//2, :]
    right_image = image[:, width//2:, :]
    return left_image, right_image

def process_image_for_yolo(image):    # 对图像进行预处理
    return image

def process_frame(frame):
    """
    将帧转换为YOLOv8模型所需的格式并进行目标检测。
    """
    results = model(frame)
    # 渲染结果
    frame_detected = results[0].plot()
    return frame_detected

def get_depth_map(left_image, right_image):
    # 计算深度图
    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    depth_map = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    # print(depth_map)
    return cv2.resize(depth_map, (320, 320))

def get_stereo_camera(name, camera_idW, resolution, color_space, fps, session):
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
                left_image_detected = process_frame(left_image_processed)

                # 将you图像送入YOLO模型进行目标检测

                right_image_processed = process_image_for_yolo(right_image)
                right_image_detected = process_frame(right_image_processed)

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
                
                cv2.imshow("Left Image with Detections", left_image_detected)

                cv2.imshow("Right Image", right_image_detected)

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
    name = "Cam1"
    camera_id = 3
    resolution = 15  # AL::kQQ720px2    15  10 or 15  Image of 640x180px
    color_space = 13  # BGR color space
    fps = 15

    get_stereo_camera(name, camera_id, resolution, color_space, fps, session)