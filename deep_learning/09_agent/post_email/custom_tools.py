from langchain.tools import tool
import  datetime
import os
# smtplib连接到邮件服务器并发送电子邮件
import smtplib
# MIMEText # MIMEText可以构建包含纯文本或HTML内容的邮件消息
from email.mime.text import MIMEText
# formataddr 函数用于格式化电子邮件地址
from email.utils import formataddr

class CustomTools():
    def __init__(self):
        pass
    @tool("将文本写入文档中")   # tool langchain的装饰器，装饰过后就是大模型可以用的function
    def write_text_to_file(content: str) -> str:
        """
            将编辑后的书信文本内容自动保存到txt文档中。
        """
        try:
            with open('./email.txt', 'w', encoding='utf-8') as f:
                f.write(content)
            return f"已写入文件：{file_path}"
        except Exception as e:
            return f"写入文件时出错：{str(e)}"