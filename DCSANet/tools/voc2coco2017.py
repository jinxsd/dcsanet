import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm


def xml_to_coco(xml_folder, image_folder, category_id_mapping):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_set = set()
    annotation_id = 1
    image_id = 1

    for xml_file in tqdm(os.listdir(xml_folder)):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        img_path = os.path.join(image_folder, filename)

        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": int(root.find('size/width').text),
            "height": int(root.find('size/height').text)
        })

        for obj in root.findall('object'):
            category_name = obj.find('name').text
            if category_name not in category_id_mapping:
                print(f"Cannot find category: {category_name}")
                continue

            category_id = category_id_mapping[category_name]
            category_set.add(category_id)

            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)

            width = x_max - x_min
            height = y_max - y_min

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    for category_name, category_id in category_id_mapping.items():
        if category_id in category_set:
            coco_data["categories"].append({
                "id": category_id,
                "name": category_name
            })

    return coco_data

def save_coco_json(coco_data, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

def convert_dataset(base_folder):
    # category_id_mapping = {
    #     "crazing": 1,
    #     "rolled-in_scale": 2,
    #     "inclusion": 3,
    #     "patches": 4,
    #     "scratches": 5,
    #     "pitted_surface": 6,
    # }
    # category_id_mapping = {
    #     "fracture": 1,
    #     "slack": 2,
    #     "wire": 3,
    #     "wear": 4,
    # }
    category_id_mapping = {
        "dent": 1,
        "erosion": 2,
        "notch": 3,
        "scratch": 4,
    }

    for split in ['train', 'val', 'test']:
        xml_folder = os.path.join(base_folder, split, 'annotations')
        image_folder = os.path.join(base_folder, split, 'images')
        coco_data = xml_to_coco(xml_folder, image_folder, category_id_mapping)

        json_path = os.path.join(base_folder, 'annotations', f'{split}.json')
        save_coco_json(coco_data, json_path)

        target_image_folder = os.path.join(base_folder, split)
        os.makedirs(target_image_folder, exist_ok=True)
        for img in coco_data['images']:
            img_src = os.path.join(image_folder, img['file_name'])
            img_dst = os.path.join(target_image_folder, img['file_name'])
            if os.path.exists(img_src):
                os.rename(img_src, img_dst)


if __name__ == "__main__":
    base_folder = '/media/amax/data/linyao/canet/data/blade_voc'
    os.makedirs(os.path.join(base_folder, 'annotations'), exist_ok=True)
    convert_dataset(base_folder)
