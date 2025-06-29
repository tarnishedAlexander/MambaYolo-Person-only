from pycocotools.coco import COCO
import os

def convert_coco_to_yolo(json_path, output_dir, img_dir):
    coco = COCO(json_path)
    os.makedirs(output_dir, exist_ok=True)
    
    person_cat_id = coco.getCatIds(catNms=['person'])[0]
    img_ids = coco.getImgIds()
    print(f"Processing {len(img_ids)} images")
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping")
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id])
        anns = coco.loadAnns(ann_ids)
        
        txt_path = os.path.join(output_dir, img_info['file_name'].replace('.jpg', '.txt'))
        with open(txt_path, 'w') as f:
            for ann in anns:
                x, y, w, h = ann['bbox']
                x_center = (x + w/2) / img_info['width']
                y_center = (y + h/2) / img_info['height']
                w = w / img_info['width']
                h = h / img_info['height']
                f.write(f"0 {x_center} {y_center} {w} {h}\n")
        if not anns:
            print(f"No annotations for {img_info['file_name']}, creating empty .txt")
            open(txt_path, 'a').close()  # Create empty .txt for images without annotations

# Convert train annotations
convert_coco_to_yolo(
    "/root/deepLearning/Mamba-YOLO/dataset/coco-person-only/images/train/_annotations.coco.json",
    "/root/deepLearning/Mamba-YOLO/dataset/coco-person-only/labels/train",
    "/root/deepLearning/Mamba-YOLO/dataset/coco-person-only/images/train"
)
# Convert valid annotations
convert_coco_to_yolo(
    "/root/deepLearning/Mamba-YOLO/dataset/coco-person-only/images/valid/_annotations.coco.json",
    "/root/deepLearning/Mamba-YOLO/dataset/coco-person-only/labels/valid",
    "/root/deepLearning/Mamba-YOLO/dataset/coco-person-only/images/valid"
)
