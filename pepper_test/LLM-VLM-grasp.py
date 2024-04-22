import cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

# 加载CLIP模型
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0是默认的摄像头ID，根据你的摄像头设置可能需要改变

try:
    while True:
        # 从摄像头读取一帧
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # 将图像从OpenCV的BGR格式转换为RGB格式
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        
        # GPT模型生成的抓取部分描述（假设已从GPT得到）
        object_part = "把手"  # 这里应该是从GPT模型动态获取
        
        # 使用CLIP处理图像和文本
        inputs = processor(text=["这是一个需要抓取的物体，抓取部分是：" + object_part], images=pil_image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        
        # 计算图像和文本的匹配概率
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        print("匹配概率:", probs.detach().numpy()[0][0])
        
        # 如果匹配概率高于某个阈值，显示匹配结果（简单示例）
        if probs > 0.5:
            cv2.putText(frame, f"Detected: {object_part}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 显示图像
        cv2.imshow('Video', frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
