from PIL import Image

color_image_path = './Data/grey.jpg'
color_image = Image.open(color_image_path)

# 将彩色图像转换为灰度图像
gray_image = color_image.convert('L')

# 保存灰度图像到文件
gray_image_path = './Data/grey_process.jpg'
gray_image.save(gray_image_path)