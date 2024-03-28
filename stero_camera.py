import time
import numpy as np
import cv2
from pepper_connection import pepper_connection

# http://doc.aldebaran.com/2-5/family/pepper_technical/video_3D_pep.html
def get_stereo_camera(name, camera_id, resolution, color_space, fps, session):
    # 订阅深度传感器
    service = session.service("ALVideoDevice")

    subscriber_id = service.subscribeCamera(name, camera_id, resolution, color_space, fps)

    try:
        # 初始化FPS计算相关变量
        frame_counter = 0
        start_time = time.time()
        while True:
            # 获取图像帧
            frame = service.getImageRemote(subscriber_id)
            
            if frame and frame[6]:
                # 提取图像数据
                width, height = frame[0], frame[1]
                array = np.frombuffer(frame[6], dtype=np.uint8)
                image = array.reshape((height, width, 3))

                # 计算并显示FPS
                frame_counter += 1
                current_time = time.time()
                if current_time - start_time > 0:
                    fps = frame_counter / (current_time - start_time)
                    cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 显示深度图像
                cv2.imshow("Stereo Camera", image)

                print("FPS: {:.2f}".format(fps))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        print("Unsubscribing from stereo camera...")
        service.unsubscribe(subscriber_id)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 连接到 Pepper
    session = pepper_connection.get_session()

    # 订阅摄像头参数
    name = "depCam1"  # name can only use 6 times
    camera_id = 3
    resolution = 15
    # Resolution name   ID    fps         Description
    # AL::k720px2	    13	10 or 15	Image of 2560x720px
    # AL::kQ720px2	    14	10 or 15	Image of 1280x360px
    # AL::kQQ720px2	    15	10 or 15	Image of 640x180px
    # AL::kQQQ720px2	16	10 or 15	Image of 320x90px
    # AL::kQQQQ720px2	17	10 or 15	Image of 160x45px
    color_space = 13  # 11:kRGBColorSpace 13:kBGRColorSpace
    fps = 15 # no need to change(10 to 15)

    get_stereo_camera(name, camera_id, resolution, color_space, fps, session)