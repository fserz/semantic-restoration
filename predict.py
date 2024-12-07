import time

import cv2
import numpy as np
from PIL import Image
import model
# from model import CYCLEGAN

if __name__ == "__main__":
    ourmodel = model.OURMODEL()
    mode = "predict"

    #video_path      = 0
    #video_save_path = ""
    #video_fps       = 25.0
    #test_interval   = 100
    #fps_image_path  = "img/1.png"

    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = ourmodel.detect_image(image)
                r_image.show()

                # 保存处理后的图像
                save_path = dir_save_path + "output_" + img.split('/')[-1]  # 保存路径
                r_image.save(save_path)  # 保存图像
                print(f"Image saved at {save_path}")
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'dir_predict'.")
