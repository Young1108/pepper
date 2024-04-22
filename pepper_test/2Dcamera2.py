import cv2
import numpy as np
from pepper_connection import pepper_connection
import time
# http://doc.aldebaran.com/2-5/family/pepper_technical/video_2D_pep.html#video-2d-pepper

def get_2D_camera(name, camera_id, resolution, color_space, fps, session):
    # 获取ALVideoDevice服务
    video_service = session.service("ALVideoDevice")

    # 先设置参数，再订阅

    # 定义自动对焦参数的ID和值
    kCameraAutoFocusID = 40
    Value = 0 # 设置为0以关闭自动对焦，设置为1以开启自动对焦

    # 设置自动对焦参数
    video_service.setParameter(camera_id, kCameraAutoFocusID, Value)

    # 设置参数并订阅摄像头
    video_client = video_service.subscribeCamera(name, camera_id, resolution, color_space, fps)

    try:
        # 初始化FPS计算
        frame_counter = 0
        start_time = time.time()
        
        while True:
            # 获取图像帧
            nao_image = video_service.getImageRemote(video_client)
            # print(nao_image)
            if nao_image is None:
                print("Failed to get image")

            # 获取图像的宽度和高度
            image_width = nao_image[0]
            image_height = nao_image[1]

            # 从返回的数据中提取图像数组
            array = nao_image[6]

            # 将图像数组转换成OpenCV的图像格式
            image = np.frombuffer(array, dtype=np.uint8).reshape((image_height, image_width, 3))

            # 将RGB图像转换为BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 计算并显示FPS
            frame_counter += 1
            current_time = time.time()
            if current_time - start_time > 0:
                fps = frame_counter /  (current_time - start_time)
                cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow("Pepper Camera", image)
            
            print("FPS: {:.2f}".format(fps))

            # 按'q'退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Unsubscribing from camera...")
        video_service.unsubscribe(video_client)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 连接到 Pepper
    session = pepper_connection.get_session()

    # 订阅摄像头参数
    name = "python_client1"
    camera_id = 0  # 0:前置摄像头, 1:下置摄像头
    resolution = 1
    # AL::k16VGA 4	fps:1 to 15	2560x1920px	Top only can't be displayed
    # AL::k4VGA	 3	fps:1 to 30 1280x960px	Top only  3
    # AL::kVGA	 2	fps:1 to 30	640x480px             10
    # AL::kQVGA	 1	fps:1 to 30	320x240px             30

    color_space = 11 # 11:RGB
    fps = 30 # no need to change

    get_2D_camera(name, camera_id, resolution, color_space, fps, session)