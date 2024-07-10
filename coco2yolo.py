import json
import os
import random
from pathlib import Path
from shutil import copyfile

# with datatset spliter
def convert_coco_to_yolo(coco_annotation_file, output_dir, img_dir, val_split=0.1):
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    categories = {category['id']: category['name'] for category in coco_data['categories']}
    annotations = coco_data['annotations']
    images = {image['id']: image for image in coco_data['images']}

    # Create output directories
    train_img_dir = os.path.join(output_dir, 'images/train')
    val_img_dir = os.path.join(output_dir, 'images/val')
    train_lbl_dir = os.path.join(output_dir, 'labels/train')
    val_lbl_dir = os.path.join(output_dir, 'labels/val')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Shuffle images and split into train and val sets
    image_ids = list(images.keys())
    random.shuffle(image_ids)
    val_size = int(len(image_ids) * val_split)
    val_image_ids = set(image_ids[:val_size])
    train_image_ids = set(image_ids[val_size:])

    for annotation in annotations:
        image_id = annotation['image_id']
        image_info = images[image_id]
        image_width = image_info['width']
        image_height = image_info['height']
        category_id = annotation['category_id']
        bbox = annotation['bbox']

        x_center = (bbox[0] + bbox[2] / 2) / image_width
        y_center = (bbox[1] + bbox[3] / 2) / image_height
        width = bbox[2] / image_width
        height = bbox[3] / image_height

        yolo_annotation = f"{category_id} {x_center} {y_center} {width} {height}\n"
        
        image_file_name = Path(image_info['file_name'].split('/')[1]).stem
        if image_id in val_image_ids:
            annotation_file = os.path.join(val_lbl_dir, f"{image_file_name}.txt")
        else:
            annotation_file = os.path.join(train_lbl_dir, f"{image_file_name}.txt")

        with open(annotation_file, 'a') as f:
            f.write(yolo_annotation)

    # Copy images to corresponding directories
    for image_id, image_info in images.items():
        image_file_name = image_info['file_name']
        src_image_path = os.path.join(img_dir, image_file_name)
        if image_id in val_image_ids:
            dest_image_path = os.path.join(val_img_dir, image_file_name.split('/')[1])
        else:
            dest_image_path = os.path.join(train_img_dir, image_file_name.split('/')[1])
        copyfile(src_image_path, dest_image_path)

if __name__ == "__main__":
    coco_annotation_file = "./UAV/annotations.json"
    output_dir = "./datasets/UAV_tune/"
    img_dir = "./UAV/"
    convert_coco_to_yolo(coco_annotation_file, output_dir, img_dir)
