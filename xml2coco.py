import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random

def convert_xml_to_coco(annotation_dir, train_output_file, test_output_file, image_dir, test_ratio=0.2):
    # Initialize COCO format dictionaries for train and test
    coco_format_train = {
        "info": {
            "year": 2021,
            "version": "1.0",
            "description": "For object detection",
            "date_created": "2021"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    coco_format_test = {
        "info": {
            "year": 2021,
            "version": "1.0",
            "description": "For object detection",
            "date_created": "2021"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}
    annotation_id = 0

    # Detect categories from XML annotations
    categories_set = set()
    xml_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.xml')])

    for filename in tqdm(xml_files, desc="Detecting categories"):
        tree = ET.parse(os.path.join(annotation_dir, filename))
        root = tree.getroot()
        for obj in root.findall('object'):
            categories_set.add(obj.find('name').text)

    # Create category mapping
    categories_list = sorted(list(categories_set))
    for idx, category in enumerate(categories_list):
        category_map[category] = idx + 1
        coco_format_train["categories"].append({"id": idx + 1, "name": category})
        coco_format_test["categories"].append({"id": idx + 1, "name": category})

    # Split data into train and test sets
    random.shuffle(xml_files)
    split_index = int(len(xml_files) * (1 - test_ratio))
    train_files = xml_files[:split_index]
    test_files = xml_files[split_index:]

    # Function to process files and ensure correct image_id mapping
    def process_files(file_list, coco_format):
        annotation_id = 0

        for filename in tqdm(file_list, desc="Processing files"):
            try:
                tree = ET.parse(os.path.join(annotation_dir, filename))
                root = tree.getroot()

                file_stem = os.path.splitext(root.find('filename').text)[0]
                try:
                    image_id = int(filename.split('.')[0]) # int(file_stem)
                    file_name = f"{image_id:05d}.png"
                except ValueError:
                    print(f"Skipping file with non-numeric name: {file_stem}")
                    continue

                image_path = os.path.join(image_dir, file_name)
                if not os.path.exists(image_path):
                    print(f"Skipping file {file_name}, corresponding image not found.")
                    continue

                # Create image info
                image_info = {
                    "height": int(root.find('size/height').text),
                    "width": int(root.find('size/width').text),
                    "date_captured": "2021",
                    "file_name": image_path,
                    "id": image_id
                }
                coco_format["images"].append(image_info)

                # Create annotations
                for obj in root.findall('object'):
                    category = obj.find('name').text
                    if category not in category_map:
                        continue

                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    width = xmax - xmin
                    height = ymax - ymin
                    area = width * height

                    annotation_info = {
                        "area": area,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": [xmin, ymin, width, height],
                        "category_id": category_map[category],
                        "id": annotation_id,
                        "occlusion": int(obj.find('truncated').text) if obj.find('truncated') is not None else 0
                    }
                    coco_format["annotations"].append(annotation_info)
                    annotation_id += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

    # Process training files
    process_files(train_files, coco_format_train)

    # Process testing files
    process_files(test_files, coco_format_test)

    # Save to json
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_file), exist_ok=True)
    
    with open(train_output_file, 'w') as f:
        json.dump(coco_format_train, f, indent=4)

    with open(test_output_file, 'w') as f:
        json.dump(coco_format_test, f, indent=4)

    print(f"Detected categories: {categories_list}")

# Usage example
xml_annotation_dir = 'Annotation'  # Replace with your annotation directory path
image_dir = 'rgb'  # Replace with your image directory path
train_output_file = 'train/_annotations.coco_back.json'  # Replace with your desired train output file path
test_output_file = 'test/_annotations.coco_back.json'  # Replace with your desired test output file path
convert_xml_to_coco(xml_annotation_dir, train_output_file, test_output_file, image_dir)
