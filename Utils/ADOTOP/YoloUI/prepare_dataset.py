import os
import shutil
import random

random.seed(42)

src_data = r"e:\jupyter file\2_Optics\8821L\Utils\YoloUI\Fore_BackGround_70_data"
dst_root = r"e:\jupyter file\2_Optics\8821L\Utils\YoloUI\Fore_BackGround_70_export"
labels_train_dir = os.path.join(dst_root, "labels", "train")

images_train_dir = os.path.join(dst_root, "images", "train")
images_val_dir = os.path.join(dst_root, "images", "val")
labels_val_dir = os.path.join(dst_root, "labels", "val")

for d in [images_train_dir, images_val_dir, labels_val_dir]:
    os.makedirs(d, exist_ok=True)

txt_files = [f for f in os.listdir(labels_train_dir) if f.endswith(".txt")]
print(f"Found {len(txt_files)} label files.")

paired = []
for txt in txt_files:
    base = os.path.splitext(txt)[0]
    src_img = os.path.join(src_data, base + ".jpg")
    if os.path.exists(src_img):
        paired.append((base, src_img))
    else:
        print(f"Warning: image not found for {base}")

print(f"Paired images: {len(paired)}")

# copy all to train first
for base, src_img in paired:
    shutil.copy2(src_img, os.path.join(images_train_dir, base + ".jpg"))

# split 8:2
random.shuffle(paired)
split_idx = int(len(paired) * 0.8)
train_list = paired[:split_idx]
val_list = paired[split_idx:]

# move val images and labels
for base, _ in val_list:
    # move image
    src_img_path = os.path.join(images_train_dir, base + ".jpg")
    dst_img_path = os.path.join(images_val_dir, base + ".jpg")
    shutil.move(src_img_path, dst_img_path)
    # copy label
    src_label = os.path.join(labels_train_dir, base + ".txt")
    dst_label = os.path.join(labels_val_dir, base + ".txt")
    shutil.copy2(src_label, dst_label)

print(f"Train: {len(train_list)}, Val: {len(val_list)}")

# update data.yaml
yaml_path = os.path.join(dst_root, "data.yaml")
yaml_content = f"""path: {dst_root.replace(os.sep, '/')}
train: images/train
val: images/val

names:
  0: Other
  1: Round
"""
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("Dataset prepared successfully!")
