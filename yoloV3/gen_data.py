import cv2
import os
def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size,expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img

if __name__ == '__main__':
    # img = cv2.imread("D:/AIstudyCode/data/yolo_data/000000004665.jpg")
    # print(img.shape)
    # new_img = cv2_letterbox_image(img,416)
    # print(new_img.shape)
    # cv2.imshow("1",new_img)
    # cv2.waitKey(0)
    dir_root = "D:/AIstudyCode/data/yolo_data"
    save_dir = "data/"
    count = 0
    for path in os.listdir(dir_root):
        print(path)
        img = cv2.imread(f"D:/AIstudyCode/data/yolo_data/{path}")
        print(img.shape)
        new_img = cv2_letterbox_image(img,416)
        print(new_img.shape)
        cv2.imwrite(f"{save_dir}images/{count}.jpg",new_img)
        count += 1
        cv2.imshow("1",new_img)
        cv2.waitKey(5)
