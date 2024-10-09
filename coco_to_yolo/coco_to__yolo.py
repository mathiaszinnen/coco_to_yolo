import argparse
from tqdm import tqdm
import random
import yaml
import json
import sys
import os
import shutil
import glob


def has_valid_imagedir(input_dir):
    image_path = f'{input_dir}/images'
    if not os.path.isdir(image_path):
        return False
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(glob.glob(f'{input_dir}/images/{ext}'))
    if len(image_files) == 0:
        return False
    return True

def has_valid_annotationdir(input_dir):
    annotations_path = f'{input_dir}/annotations'
    if not os.path.isdir(annotations_path):
        return False
    if len(glob.glob(f'{input_dir}/annotations/*.json')) != 1:
        # existing splits might be implemented later
        return False
    return True

def validate_input(input_dir):
    if not has_valid_imagedir(input_dir):
        print("Please provide a valid image directory with at least one input image (jpg, jpeg, bmp).")
        sys.exit(1)
    if not has_valid_annotationdir(input_dir):
        print("Please provide a valid annotations directory with at exactly one COCO annotations file (json).")
        sys.exit(1)

def create_yolo_structure(output_dir, name, test_ratio, val_ratio):
    os.makedirs(f'{output_dir}/{name}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/{name}/labels/train', exist_ok=True)
    if test_ratio > 0.0:
        os.makedirs(f'{output_dir}/{name}/images/test', exist_ok=True)
        os.makedirs(f'{output_dir}/{name}/labels/test', exist_ok=True)
    if val_ratio > 0.0:
        os.makedirs(f'{output_dir}/{name}/images/valid', exist_ok=True)
        os.makedirs(f'{output_dir}/{name}/labels/valid', exist_ok=True)

def create_yaml(output_dir, name, test_ratio, val_ratio, classes):
    dataset_dict = {
        'path': '.',
        'train': 'images/train'
    }
    if test_ratio > 0.0:
        dataset_dict['test'] = 'images/test'
    if val_ratio > 0.0:
        dataset_dict['val'] = 'images/valid'
    class_dict = {}
    for i, cls in enumerate(classes):
        class_dict[i] = cls
    dataset_dict['names'] = class_dict
    with open(f'{output_dir}/{name}.yaml', 'w') as f:
        yaml.dump(dataset_dict,f) 

def create_splits(ids, test_ratio, val_ratio):
    test_ids = []
    val_ids = []
    n_test = int(test_ratio * len(ids))
    n_val = int(val_ratio * len(ids))
    test_ids = random.choices(ids, k=n_test)
    train_ids = [id for id in ids if id not in test_ids]
    val_ids = random.choices(train_ids, k=n_val)
    train_ids = [id for id in train_ids if id not in val_ids]
    return train_ids, test_ids, val_ids

def copy_images(input_dir,output_dir,name,ids, split, img_id_map):
    print(f'Copying {split} images...')
    for id in tqdm(ids):
        fn = img_id_map[id]
        shutil.copyfile(f'{input_dir}/images/{fn}', f'{output_dir}/{name}/images/{split}/{fn}')    

def to_yolo_bbox(bbox,im_w, im_h):
    x,y,w,h = bbox
    mid_x_rel = (x + w/2) / im_w
    mid_y_rel = (y + h/2) / im_h
    w_rel = w / im_w
    h_rel = h / im_h
    return [mid_x_rel, mid_y_rel, w_rel, h_rel]

def create_annotation_map(coco):
    img_id_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id in img_id_to_anns.keys():
            img_id_to_anns[img_id].append(ann)
        else:
            img_id_to_anns[img_id] = [ann]
    return img_id_to_anns

def create_annotations(output_dir,name,split,coco,img_ids):
    img_id_to_anns = create_annotation_map(coco)

    img_id_to_imgs = {img['id']: img for img in coco['images']}
    print(f'Creating {split} labels...')
    for img_id in tqdm(img_ids):
        img = img_id_to_imgs[img_id]
        try:
            anns = img_id_to_anns[img_id]
        except KeyError: # no annotations for image -> continue
            continue
        width = img['width']
        height = img['height']
        ann_strings = []
        for ann in anns:
            cat_id = ann['category_id'] - 1 # assuming annotations are consecutively indexed 
            yolo_bbox = to_yolo_bbox(ann['bbox'],width,height)
            bbox_string = ' '.join([str(x) for x in yolo_bbox])
            ann_strings.append(f'{cat_id} {bbox_string}')

        img_ann_strings = ' '.join(ann_strings)
        image_identifier = os.path.splitext(img['file_name'])[0]
        output_path = f'{output_dir}/{name}/labels/{split}/{image_identifier}.txt'
        with open(output_path, 'w') as f:
            f.write(img_ann_strings)

def convert(args):
    annotations_path = glob.glob(f'{args.input_dir}/annotations/*.json')[0]
    with open(annotations_path) as f:
        coco = json.load(f)
    classes = [cat['name'] for cat in coco['categories']] 

    create_yaml(args.output_dir, args.dataset_name, args.test_ratio, args.val_ratio, classes)

    imgids = [img['id'] for img in coco['images']]
    split_ids = create_splits(imgids, args.test_ratio, args.val_ratio)
    img_id_to_fn = {img['id']: img['file_name'] for img in coco['images']}
    for img_ids, split in zip(split_ids, ['train', 'test', 'valid']):
        copy_images(args.input_dir,args.output_dir,args.dataset_name, img_ids, split, img_id_to_fn)
        create_annotations(args.output_dir,args.dataset_name,split,coco,img_ids)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert COCO datasets to YOLO format.'
    )
    parser.add_argument('input_dir', help='Path to source COCO dataset directory.')
    parser.add_argument('output_dir', help='Output directory to store converted YOLO dataset.')
    parser.add_argument('--dataset_name', help='Dataset name to use for YOLO formatted output.', required=False, default='converted')
    parser.add_argument('--test_ratio', help='Ratio of the data to be splitted for the test set. Please provide a number between 0.0 and 1.0', required=False, type=float, default=0.1)
    parser.add_argument('--val_ratio', help='Ratio of the data to be splitted for the validation set. Please provide a number between 0.0 and 1.0', required=False, type=float, default=0.0)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    validate_input(args.input_dir)
    create_yolo_structure(args.output_dir, args.dataset_name, args.test_ratio, args.val_ratio)
    convert(args)

if __name__ == '__main__':
    main()
