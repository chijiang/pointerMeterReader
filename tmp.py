import os


pics_dir = os.listdir("data/detection_yolo/images/train")

pics = [x.split(".")[0] for x in pics_dir]

label_dir = os.listdir("data/detection_yolo/labels/train")

label_files = [x.split(".")[0] for x in label_dir]

for label in label_files:
    if label not in pics:
        os.remove(f"data/detection_yolo/labels/train/{label}.txt")

        