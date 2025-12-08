import os
import numpy as np
import cv2





image_folder = "raw_data/images/"

image_names = os.listdir(image_folder)


category = input("specify category: ")
category_dir = os.path.join("raw_data/data", category)

if not os.path.exists(category_dir):
    os.mkdir(category_dir)


start_pos = None
end_pos = None
def mouse_press(event, x, y, flags, param):
    global start_pos, end_pos, image, image_name
    if event == cv2.EVENT_LBUTTONDOWN:
        start_pos = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        end_pos = x, y
        slices = slice(start_pos[1], end_pos[1]), slice(start_pos[0], end_pos[0])
        image_slice = image[slices]
        cv2.imwrite(os.path.join(category_dir, f"{image_name.split(".")[0]}_{os.urandom(4).hex()}.jpg"), image_slice)
        print("saved image")

for image_name in image_names:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    H, W, C = image.shape
    aspect_ratio = W/H
    W = min(W, 1920)
    H = min(H, 1080)
    W = min(W, int(H/aspect_ratio))
    H = min(W, int(W*aspect_ratio))
    image = cv2.resize(image, (W, H))
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", mouse_press)
    key = cv2.waitKey() & 0xff
    if key == 27: # esc
        break






