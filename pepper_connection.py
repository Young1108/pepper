import qi
import sys

class PepperConnection:
    def __init__(self, robot_ip, robot_port=9559):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.app = None
        self.session = None
        self.connect()

    def connect(self):
        """连接到Pepper机器人"""
        try:
            self.app = qi.Application(["RobotApp", "--qi-url=tcp://" + self.robot_ip + ":" + str(self.robot_port)])
            self.app.start()
            self.session = self.app.session
            print(f"成功连接到 Pepper @{self.robot_ip}:{self.robot_port}")
        except RuntimeError as e:
            print(f"连接Pepper时发生错误: {e}")
            sys.exit(1)

    def get_session(self):
        """返回当前的会话对象"""
        if self.session is not None:
            print("Session created.")
            return self.session
        else:
            print("当前没有连接到 Pepper。")
            return None

# 创建全局连接实例
ip_address = "10.151.15.237"
pepper_connection = PepperConnection(ip_address)
