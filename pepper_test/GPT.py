import qi
from openai import OpenAI
from pepper_connection import pepper_connection
import os
from dotenv import load_dotenv

# os.environ["http_proxy"] = "45.78.59.189"
# os.environ["https_proxy"] = "45.78.59.189"
# 从.env文件中加载环境变量
load_dotenv()

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
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    # 提取并返回最后一条消息的内容
    return completion.choices[0].message.content

def main():
    session = pepper_connection.get_session()

    # 使用session对象来创建服务实例
    tts = session.service("ALTextToSpeech") # Text to speech
    memory = session.service("ALMemory")
    dialog = session.service("ALDialog")
    dialog.setLanguage("English")

    # 启动对话模块
    dialog.subscribe('my_dialog_example')

    print("Pepper is ready to chat! Say 'bye' to end the conversation.")

    try:
        while True:
            # 检测用户的说话内容
            try:
                # Pepper机器人等待说话的关键字
                qi_chat_variable = "Dialog/LastInput"
                user_said = memory.getData()
                print(f"[Debug] Raw user said: {user_said}")
            except Exception as e:
                print(f"Error getting user input: {e}")
                user_said = None
            if user_said:
                print(f"User said: {user_said}")

            # 如果用户说了"bye"，则结束对话
            if user_said.strip().lower() == "bye":
                print("Ending conversation.")
                tts.say("Goodbye!")
                break

            # 使用OpenAI GPT模型获取回复
            gpt_response = ask_gpt(user_said)
            print(f"GPT: {gpt_response}")

            # 使用Pepper的语音服务朗读回复
            tts.say(gpt_response)
    finally:
        # 取消对话订阅以清理资源
        dialog.unsubscribe('my_dialog_example')

if __name__ == "__main__":
    main()
