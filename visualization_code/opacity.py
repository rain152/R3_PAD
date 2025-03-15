from rembg import remove
from PIL import Image
import os

def remove_background(input_path, output_path):
    """
    去除图像背景并保存结果。

    参数：
    - input_path (str): 输入图像路径。
    - output_path (str): 输出图像路径。
    """
    # 打开输入图像
    with open(input_path, 'rb') as input_file:
        input_image = input_file.read()
    
    # 去除背景
    output_image = remove(input_image)
    
    # 保存输出图像
    with open(output_path, 'wb') as output_file:
        output_file.write(output_image)

# 示例用法
if __name__ == "__main__":
    input_image_path = "图片3.png"  # 输入图像路径
    output_image_path = input_image_path  # 输出图像路径（建议保存为 PNG 格式以支持透明背景）
    
    remove_background(input_image_path, output_image_path)
    print(f"背景已去除，结果保存在 {output_image_path}")