import csv
import os
import shutil
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--classes', nargs='+', type=str)
args = parser.parse_args()

classes = args.classes

print(classes)

src_dir="OID/Dataset/"

object_detection_dst_dir="../object_detection/train/"
object_detection_img_dst_dir=object_detection_dst_dir+"images/"
object_detection_csv_dst_file=object_detection_dst_dir+"annotations.csv"

image_classification_dst_dir="../image_classifier/train/data_set/"

if not os.path.exists(object_detection_img_dst_dir):
    os.makedirs(object_detection_img_dst_dir)

if not os.path.exists(object_detection_dst_dir):
    os.makedirs(object_detection_dst_dir)

with open(object_detection_csv_dst_file, 'w') as csv_dst:

    data = []
    
    for dir_name in ["test", "validation", "train"]:
        print("Processing data for "+dir_name)
        dir_path = os.path.join(src_dir, dir_name)
      
        for subdir_name in tqdm(os.listdir(dir_path)):
            subdir_path = os.path.join(dir_path, subdir_name)
            label_path = os.path.join(subdir_path, "Label")

            label = os.path.basename(subdir_path)

            if label.replace(' ', '_') not in classes:
                continue

            image_classification_class_dst_dir =  os.path.join(image_classification_dst_dir, label)
            
            if not os.path.exists(image_classification_class_dst_dir):
                os.makedirs(image_classification_class_dst_dir)

            if os.path.isdir(label_path):
                with open(os.path.join(label_path, "annotations.csv"), mode ='r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        f = dir_name.upper() +',images/'+row[0]+','+subdir_name+','+ ','.join(row[2:])
                        data.append(f)
                
                for file_name in os.listdir(subdir_path):
                    if file_name.endswith(".jpg"):
                        file_path = os.path.join(subdir_path, file_name)
                        
                        # uncomment below if you don't need the images for object detection
                        shutil.copy2(file_path, object_detection_img_dst_dir)

                        # uncomment below if you don't need the images for image classification
                        shutil.copy2(file_path, image_classification_class_dst_dir)

    
    writer = csv.writer(csv_dst)
    for row in data:
        writer.writerow(row.split(","))

