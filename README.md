# coco-to-yolo
Simple command line tool to convert COCO object detection datasets to YOLO format.

## Usage
1. Install via pip `pip install coco-to-yolo`
2. Convert COCO dataset to [ultralytics](https://docs.ultralytics.com/) YOLO format using `coco_to_yolo <<coco_dir>> <<output_dir>>`

By default the script assumes the coco dataset to be structured as follows:

    <<coco_dir>>
    ├── annotations
    │   └── annotations.json     # Exactly ONE annotation file in COCO json format
    └── images                   # Arbitary number of images (matching the file names 
        ├── image1.jpeg          # in the annotations json file)
        ├── image2.jpeg         
        └── ...

## Split generation
By default the script will split 10% of the data into a test split and not generate a validation split. The ratio of splitted test and validation data can be adapted by specifying the `--test_ratio` and `--val_ratio` arguments, e.g. 

## Example usage
```
coco_to_yolo /home/COCO_ds /home/COCO_ds --test_ratio 0.15 --val_ratio 0.1
```
will convert the dataset in the `/home/COCO_ds` to the format required by YOLO, split 15% of the data for the testing, 10% for validation, and store the resulting dataset in `/home/COCO_ds` 

## Contributions
Feel free to suggest extensions and point out mistakes by creating an issue or sending me a pull request.

