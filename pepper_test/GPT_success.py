import qi
import os
from dotenv import load_dotenv
import speech_recognition as sr
from openai import OpenAI
from pepper_connection import pepper_connection
import pyaudio


# 从.env文件中加载环境变量
load_dotenv()
# 设置OpenAI客户端，配置API密钥
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url="https://key.wenwen-ai.com/v1"
# )
# 设置OpenAI客户端，配置API密钥
client = OpenAI(
    # api_key= "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",
    # base_url="45.78.59.189"
    api_key = "sk-NYsoG3VBKDiTuvdtC969F95aFc4f45379aD3854a93602327",
    base_url="https://key.wenwen-ai.com/v1"
)

def ask_gpt(prompt):
    """使用OpenAI GPT模型生成回复"""
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are the robot of SOftBank company. Your name is Pepper, which is a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    # 提取并返回最后一条消息的内容
    return completion.choices[0].message.content

def recognize_speech_from_mic(recognizer, microphone):
    """使用麦克风捕获语音并识别"""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # 使用Google Web Speech API进行识别
    try:
        return recognizer.recognize_google(audio)
    except sr.RequestError:
        # API无法到达
        return "API unavailable"
    except sr.UnknownValueError:
        # 无法识别语音
        return "Unable to recognize speech"

def main(session):
    tts = session.service("ALTextToSpeech")

    recognizer = sr.Recogn0izer()
    microphone = sr.Microphone()

    print("Pepper is ready to chat! Say 'bye' to end the conversation.")

    while True:
        print("Listening...")
        text = recognize_speech_from_mic(recognizer, microphone)
        print("You said:", text)

        # 检查是否是结束对话的指令
        if text.lower() == "bye":
            print("Ending conversation.")
            tts.say("Goodbye!")
            break

        # 否则，使用ChatGPT获取回复
        response = ask_gpt(text)
        print("GPT:", response)
        
        # 使用Pepper的TTS服务朗读回复
        tts.say(response)

if __name__ == "__main__":
    session = pepper_connection.get_session()
    main(session)
# command: /bin/python3 /home/huangjiayang/code/GPT_test.py 2>/dev/null