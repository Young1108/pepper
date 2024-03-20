import time
import numpy as np
import cv2
from pepper_connection import pepper_connection
# http://doc.aldebaran.com/2-5/family/pepper_technical/video_3D_pep.html

def get_3D_camera(name, camera_id, resolution, color_space, fps, session):
    # 订阅深度传感器
    depth_service = session.service("ALVideoDevice")

    subscriber_id = depth_service.subscribeCamera(name, camera_id, resolution, color_space, fps)

    try:
        # 初始化FPS计算相关变量
        frame_counter = 0
        start_time = time.time()
        while True:
            # 获取深度图像帧
            depth_frame = depth_service.getImageRemote(subscriber_id)

            if depth_frame and depth_frame[6]:
                # 提取图像数据
                width, height = depth_frame[0], depth_frame[1]
                depth_array = np.frombuffer(depth_frame[6], dtype=np.uint16)  # 每个深度值是2字节
                depth_image = depth_array.reshape((height, width))  # 对于2字节的深度值，不需要第三维

                # 转换为可视化的灰度图像
                depth_image_visual = cv2.convertScaleAbs(depth_image, alpha=0.08)  # alpha调节以获得最佳视觉效果
                
                # 计算并显示FPS
                frame_counter += 1
                current_time = time.time()
                if current_time - start_time > 0:
                    fps = frame_counter / (current_time - start_time)
                    cv2.putText(depth_image_visual, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 显示深度图像
                cv2.imshow("Pepper Depth Camera", depth_image_visual)

                print("FPS: {:.2f}".format(fps))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        print("Unsubscribing from depth camera...")
        depth_service.unsubscribe(subscriber_id)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 连接到 Pepper
    session = pepper_connection.get_session()

    # 订阅摄像头参数
    name = "depCam"  # name can only use 6 times
    camera_id = 2
    resolution = 9  # 5:1280*720px, 9:640*480px, 10:320*240px, 11:160*120px
    color_space = 17  # 0:kYuvColorSpace 11:kRGB 17:kDepth
    fps = 15 # no need to change(10 to 15)

    get_3D_camera(name, camera_id, resolution, color_space, fps, session)