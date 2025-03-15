
import os
from PIL import Image

# 定义文件夹路径和类别名
base_dir = "./Draw"
folders = ["ori", "mask", "L0", "L1", "L2", "L3", "ours"]
categories = [
    "01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "12Zalika", "17Scorpion"
]  # 示例类别

# 定义图片之间的间隔（5像素）
spacing = 5

# 目标图片大小
target_size = (224, 224)

# 遍历每个类别
for category in categories:
    # 获取该类别下的所有图片索引
    image_indices = sorted(
        [int(f.split(".")[0]) for f in os.listdir(os.path.join(base_dir, folders[0], category))]
    )

    # 遍历每张图片
    for idx in image_indices:
        # 存储当前图片的路径
        image_paths = [
            os.path.join(base_dir, folder, category, f"{idx}.png") for folder in folders
        ]

        # 打开所有图片并 resize 到 224x224
        images = [Image.open(img_path).resize(target_size) for img_path in image_paths]

        # 计算拼接后的大图宽度
        total_width = sum(img.width for img in images) + (len(images) - 1) * spacing

        # 创建一个新的空白图片
        max_height = max(img.height for img in images)
        combined_image = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

        # 拼接图片
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width + spacing

        # 保存拼接后的图片
        output_dir = os.path.join(base_dir, "combined", category)
        os.makedirs(output_dir, exist_ok=True)
        combined_image.save(os.path.join(output_dir, f"combined_{idx}.png"))

        print(f"Processed {category}/{idx}.png")

print("All images processed!")