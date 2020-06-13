from xml.dom.minidom import parse
import os

train_label = r"data/train_label.txt"
train_label_txt = open(train_label,"w")
dir_root = "data/outputs/"

for path in os.listdir(dir_root):
    #print(dir_root)
    xml_doc = dir_root+path
    print(xml_doc)

    #xml_doc = r"D:/AIstudyCode/data/yolo_data/outputs/000000003590.xml"
    dom = parse(xml_doc) # 读取文件
    root = dom.documentElement # root为xml文件得根节点
    # print(dom)
    # print(root)
    # root。 根节点下得…结点
    img_name = root.getElementsByTagName("path")[0].childNodes[0].data #文件名结点

    img_size= root.getElementsByTagName("size")[0] # 文件尺寸结点
    objects = root.getElementsByTagName("object") # 标注信息结点
    #print(objects[0].getElementsByTagName("name"))
    #exit()
    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    print(img_name)
    print(img_w,img_h,img_c)
    #for box in objects[0].childNodes[0].data.getElementsByTagName("item")[0]:
    #    print(box)
    train_label_txt.write(f"{img_name} ")
    train_label_txt.flush()
    #exit()
    for box in objects:
        for i in range(len(box.getElementsByTagName("name"))):
            cls_name = box.getElementsByTagName("name")[i].childNodes[0].data
            if cls_name == "person": tag = 0
            elif cls_name == "Motor": tag = 1
            elif cls_name == "plane": tag = 2
            elif cls_name == "car": tag = 3
            elif cls_name == "dog": tag = 4
            x1 = int(box.getElementsByTagName("xmin")[i].childNodes[0].data)
            y1 = int(box.getElementsByTagName("ymin")[i].childNodes[0].data)
            x2 = int(box.getElementsByTagName("xmax")[i].childNodes[0].data)
            y2 = int(box.getElementsByTagName("ymax")[i].childNodes[0].data)
            print(tag,x1,y1,x2,y2)

            cx, cy = (x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1
            w, h = x2 - x1, y2 - y1

            train_label_txt.write(f"{tag} {cx} {cy} {w} {h} ")
            train_label_txt.flush()
    train_label_txt.write(f"\n")
    train_label_txt.flush()






