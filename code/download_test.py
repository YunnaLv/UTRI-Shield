import os

from PIL import Image
from torchvision.utils import save_image
import shutil

# 定义下载图片的函数
def download_images_from_txt(txt_file, output_dir):
    # 创建目标文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开txt文件，逐行读取图片URL并下载
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # 去除行尾的换行符
            line = line.strip()
            # 以空格分割每行内容，第一个元素是图片URL
            parts = line.split(' ')
            # print(parts)
            # break
            image_url = parts[0]
            # 提取图片文件名
            image_filename = '/data/UTAH_datasets/CASIA-WebFace/' + image_url
            # print(image_filename)
            # break
            # 拼接保存路径
            save_path = output_dir + '/' + image_url.split('/')[0] + '/'
            print(save_path)
            # 保存图像
            # save_image(Image.open(image_filename), save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            shutil.copy(image_filename, save_path)

# 调用函数下载图片到指定文件夹
txt_file = './data/CASIA/OSN.txt'  # 替换为你的txt文件路径
output_folder = 'download_test'  # 替换为你希望保存图片的文件夹路径
download_images_from_txt(txt_file, output_folder)
