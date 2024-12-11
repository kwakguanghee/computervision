import os.path
import argparse
import json
from PIL import Image
import requests
from io import BytesIO
import sys

parser = argparse.ArgumentParser(description='Download images and annotations for TACO dataset')
parser.add_argument('--dataset_path', required=False, default='./data/annotations.json', help='Path to annotations')
parser.add_argument('--output_path', required=False, default='./dataset/yolo', help='Path to save YOLO formatted dataset')
args = parser.parse_args()

dataset_dir = os.path.dirname(args.dataset_path)
output_dir = args.output_path

# Create output directories
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

print('Note. If for any reason the connection is broken. Just call me again and I will start where I left.')

# Load annotations
with open(args.dataset_path, 'r') as f:
    annotations = json.loads(f.read())

    nr_images = len(annotations['images'])
    for i in range(nr_images):

        image = annotations['images'][i]

        file_name = image['file_name']
        url_original = image['flickr_url']
        url_resized = image['flickr_640_url']

        file_path = os.path.join(output_dir, 'images', file_name)

        # Create subdir if necessary
        subdir = os.path.dirname(file_path)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        if not os.path.isfile(file_path):
            # Load and Save Image
            response = requests.get(url_original)
            img = Image.open(BytesIO(response.content))
            if img._getexif():
                img.save(file_path, exif=img.info["exif"])
            else:
                img.save(file_path)

        # Save YOLO format annotations
        annotations_file = os.path.join(output_dir, 'labels', file_name.replace('.jpg', '.txt'))  # Assuming images are .jpg

        # Ensure the directory for labels exists
        label_dir = os.path.dirname(annotations_file)
        os.makedirs(label_dir, exist_ok=True)  # Create the label directory if it doesn't exist

        # Get the annotations for this image
        image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image['id']]

        with open(annotations_file, 'w') as label_file:
            for ann in image_annotations:
                # YOLO format: class_id x_center y_center width height
                bbox = ann['bbox']  # [x, y, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / image['width']
                y_center = (bbox[1] + bbox[3] / 2) / image['height']
                width = bbox[2] / image['width']
                height = bbox[3] / image['height']

                # Get class_id
                class_id = ann['category_id']  # Assuming category_id directly maps to YOLO class id
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        # Show loading bar
        bar_size = 30
        x = int(bar_size * i / nr_images)
        sys.stdout.write("%s[%s%s] - %i/%i\r" )
