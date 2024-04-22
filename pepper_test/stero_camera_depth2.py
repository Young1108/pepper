import cv2
import numpy as np
from pepper_connection import pepper_connection

def get_depth_map(imgL, imgR):
    # 创建立体匹配对象
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # 计算两个图像的视差图
    disparity = stereo.compute(imgL, imgR)
    # 归一化视差图以便显示
    # _, disparity = cv2.threshold(disparity, 0, 255, cv2.THRESH_TOZERO)
    _, disparity = cv2.threshold(disparity, 0, 255, cv2.THRESH_TOZERO)
    disparity_normalized = (disparity / 16.).astype(np.uint8)
    # 生成伪彩色深度图
    depth_map = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    return depth_map

def process_stereo_images(image):
    # 分割图像
    height, width, _ = image.shape
    width_cutoff = width // 2
    imgL = image[:, :width_cutoff]
    imgR = image[:, width_cutoff:]
    # 转换为灰度
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # 获取深度图
    depth_map = get_depth_map(imgL_gray, imgR_gray)
    return imgL, imgR, depth_map

# 连接到 Pepper
session = pepper_connection.get_session()

# 订阅摄像头参数
name = "StereoCam"
camera_id = 3  # 0是顶部的摄像头，1是底部的摄像头，3是立体视觉
resolution = 15  # VGA
color_space = 13  # RGB
fps = 15

# 创建视频设备服务实例
video_service = session.service("ALVideoDevice")
subscriber_id = video_service.subscribeCamera(name, camera_id, resolution, color_space, fps)

try:
    while True:
        # 获取图像帧
        frame = video_service.getImageRemote(subscriber_id)
        if frame and frame[6]:
            # 提取图像数据
            width, height = frame[0], frame[1]
            array = np.frombuffer(frame[6], dtype=np.uint8)
            image = array.reshape((height, width, 3))

            # 分割并处理立体图像以获取深度信息
            imgL, imgR, depth_map = process_stereo_images(image)

            # 显示结果
            cv2.imshow("Left Camera", imgL)
            cv2.imshow("Right Camera", imgR)
            cv2.imshow("Depth Map", depth_map)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    print("Unsubscribing from stereo camera...")
    video_service.unsubscribe(subscriber_id)
    cv2.destroyAllWindows()
