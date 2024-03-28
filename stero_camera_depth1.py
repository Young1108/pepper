import time
import numpy as np
import cv2
from pepper_connection import pepper_connection

# 立体匹配器设置
# 这里使用的是SGBM（Semi-Global Block Matching）算法 立体匹配算法
# 更复杂的环境可能需要调整这些参数
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,    # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * 5 ** 2,    # 8*number_of_image_channels*SADWindowSize*SADWindowSize
    P2=32 * 3 * 5 ** 2,   # 32*number_of_image_channels*SADWindowSize*SADWindowSize
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

def get_stereo_images(image, width):
    # 将图像分为左右两个部分
    left_image = image[:, :width//2, :]
    right_image = image[:, width//2:, :]
    return left_image, right_image

def get_depth_map(left_image, right_image):
    # 计算深度图
    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    depth_map = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return depth_map

def get_stereo_camera(name, camera_id, resolution, color_space, fps, session):
    # 订阅深度传感器
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

                cv2.imshow("Left Image", left_image)
                cv2.imshow("Right Image", right_image)
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
    name = "depCam3"
    camera_id = 3
    resolution = 15  # QQVGA resolution
    color_space = 13  # BGR color space
    fps = 15

    get_stereo_camera(name, camera_id, resolution, color_space, fps, session)
