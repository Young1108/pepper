import os
from openai import OpenAI


# 设置OpenAI客户端，配置API密钥
client = OpenAI(
    api_key="sk-UBDgT2rLkYLWKOTXp9voiO8WfnQxzVWvQ6yGSIA1ZUDxtr47",
    base_url="https://api.chatanywhere.tech/v1"
)


def ask_gpt(prompt):
    """使用OpenAI GPT模型生成回复"""
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are an intelligent robotic arm. You will receive the name of an object from a user. "
                        "If you want to pick up this object, which part makes the most sense to grasp? Please reply one word of the grasping part in your feedback"},
            {"role": "user", "content": prompt},
        ]
    )
    # 提取并返回最后一条消息的内容
    return completion.choices[0].message.content


def main():
    print("Type 'bye' to end the conversation.")
    while True:
        object_name = input("Enter the name of the object you want to pick up: ")

        if object_name.lower() == "bye":
            print("Ending conversation.")
            break

        # 使用ChatGPT获取回复
        response = ask_gpt(object_name)
        print("GPT:", response)


if __name__ == "__main__":
    main()
