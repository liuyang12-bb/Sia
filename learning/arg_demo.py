import argparse

# 1. 创建参数解析器
parser = argparse.ArgumentParser()

# 2. 告诉程序允许接收什么参数
parser.add_argument("--name", type=str, default="Tom")
parser.add_argument("--age", type=int, default=18)

# 3. 解析命令行传来的参数
args = parser.parse_args()

# 4. 使用参数
print("姓名：", args.name)
print("年龄：", args.age)