#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


'''
定义一个简单的一维卡尔曼滤波器，用于估计一个物体的高度
物体初始高度为100m，速度为0，加速度为9.8m/s^2
每秒钟采样1000次.观测值的方差为1，过程噪声的方差为0.01，观测噪声的方差为4
'''
#定义步长和迭代次数
iterations = 1000
delta_t = 0.001
#生成时间序列
time = np.linspace(0,1,iterations)
time = np.mat(time)

#定义g值
g = 9.8

# 系统真值
z = [100-0.5*g*(delta_t*i)**2 for i in range(iterations)]
z_watch = np.mat(z)

# 高斯噪声是一个均值为0的高斯分布，方差为1
# 创建一个方差为1的高斯噪声，精确到小数点后两位
noise = np.round(np.random.normal(0, 1, iterations), 2)
noise_mat = np.mat(noise) 

# 将z的观测值和噪声相加
z_mat = z_watch + noise_mat

# 定义最优估计的输出
y = []

# 定义x的初始状态
h0 = 105 # 为了更好的观察效果，将初始高度设置的比真实值高
v0 = 0  # 初始速度
x_mat = np.mat([[h0,], [v0,]])
# 定义初始状态协方差矩阵
p_mat = np.mat([[10, 0], [0, 0.01]])
# 定义状态转移矩阵，因为每秒钟采样1000次，所以delta_t = 0.001
f_mat = np.mat([[1, delta_t], [0, 1]])
# 定义输入矩阵
g_mat = np.mat([[-0.5*delta_t**2],[-delta_t]])
# 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
q_mat = np.mat([[0.0, 0], [0, 0.0]])
# 定义观测矩阵
h_mat = np.mat([1, 0])
# 定义观测噪声协方差
r_mat = np.mat([4])

#卡尔曼滤波器的5个公式
for i in range(iterations):
    x_predict = f_mat * x_mat + g_mat * g
    p_predict = f_mat * p_mat * f_mat.T + q_mat
    kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
    x_mat = x_predict + kalman *(z_mat[0, i] - h_mat * x_predict)
    p_mat = (np.eye(2) - kalman * h_mat) * p_predict
    
    #将每步计算结果添加到序列中
    y.append(x_mat[0].tolist()[0][0])

#数据格式转化
y = np.mat(y)
error = y - z_watch
y = y.A
error = error.A
time = time.A
z_mat = z_mat.A
z_watch = z_watch.A

# 绘制实际高度、测量高度和卡尔曼滤波器估计的高度随时间的变化图
# 测量高度是真实高度加上高斯噪声
# 估计高度是卡尔曼滤波器的输出
plt.figure(1)
plt.plot(time[0,:], z_mat[0,:],  'b',   label = 'Measured')
plt.plot(time[0,:], z_watch[0,:],'g',   label='True')
plt.plot(time[0,:], y[0,:],      'r',   label = 'Estimated')
plt.xlabel('h(m)')
plt.ylabel('time(s)')
plt.legend(loc='lower right')

# 绘制误差图，观察误差变化
plt.figure(2)
plt.plot(time[0,:], error[0,:], label ='Errors')
plt.ylabel('error(m)')
plt.xlabel('time(s)')
plt.legend(loc='lower right')
plt.show()
