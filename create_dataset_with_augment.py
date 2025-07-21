import os
import shutil
import random
from collections import defaultdict
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import csv

# ---------------------------------------
# CONFIGURATION DES CHEMINS ET PARAMÈTRES
# ---------------------------------------

base_images_dir = "dataset/images"
base_labels_dir = "dataset/labels"
output_base = "dataset_balanced"

for split in ["train", "val", "test"]:
    os.makedirs(f"{output_base}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_base}/labels/{split}", exist_ok=True)

split_targets = {"train": 48000, "val": 16000, "test": 16000}
class_ratios = {1: 0.30, 0: 0.25, 2: 0.20, 4: 0.15, 3: 0.10}  # sheep, person, wolf, dog, horse

MAX_IMAGES_PER_VIDEO = 3


def get_video_id(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]


def apply_augmentations(img):
    return [
        ImageEnhance.Contrast(img).enhance(1.3),
        ImageOps.mirror(img),
        ImageOps.autocontrast(img),
        img.filter(ImageFilter.SHARPEN)
    ]


def collect_all_class_images():
    full_class_to_images = defaultdict(list)
    for split_name in ["train", "val", "test"]:
        image_dir = os.path.join(base_images_dir, split_name)
        label_dir = os.path.join(base_labels_dir, split_name)
        for img_file in os.listdir(image_dir):
            if not img_file.endswith(".jpg"):
                continue
            lbl_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))
            if not os.path.exists(lbl_path):
                continue
            with open(lbl_path, "r") as f:
                for line in f:
                    cls = int(line.strip().split()[0])
                    if cls in class_ratios:
                        full_class_to_images[cls].append((img_file, lbl_path, image_dir))
    return full_class_to_images


all_class_source_images = collect_all_class_images()


def process_split(split_name):
    target = split_targets[split_name]
    class_distribution = {cls: int(ratio * target) for cls, ratio in class_ratios.items()}
    allowed_classes = set(class_distribution.keys())

    class_to_images = defaultdict(list)
    video_usage_count = defaultdict(int)
    image_to_classes = dict()
    class_final_count = defaultdict(int)

    image_dir = os.path.join(base_images_dir, split_name)
    label_dir = os.path.join(base_labels_dir, split_name)

    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.endswith(".jpg"):
            continue
        lbl_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path, "r") as f:
            classes = set()
            for line in f:
                cls = int(line.strip().split()[0])
                if cls in allowed_classes:
                    classes.add(cls)
        if not classes:
            continue
        video_id = get_video_id(img_file)
        image_to_classes[img_file] = classes
        for cls in classes:
            class_to_images[cls].append((img_file, lbl_path, video_id))

    used_images = set()
    image_final_set = []

    for cls, target_count in class_distribution.items():
        selected = 0
        candidates = class_to_images[cls]
        random.shuffle(candidates)

        for img_file, lbl_file, video_id in candidates:
            if img_file in used_images:
                continue
            if video_usage_count[video_id] >= MAX_IMAGES_PER_VIDEO:
                continue
            used_images.add(img_file)
            video_usage_count[video_id] += 1
            image_final_set.append((img_file, lbl_file, image_dir))
            class_final_count[cls] += 1
            selected += 1
            if selected >= target_count:
                break

        needed = target_count - class_final_count[cls]
        if needed > 0:
            backup_candidates = [item for item in class_to_images[cls] if item[0] not in used_images]
            random.shuffle(backup_candidates)
            extra_originals = int(0.3 * needed)
            extra_augmented = needed - extra_originals

            for i in range(min(extra_originals, len(backup_candidates))):
                img_file, lbl_file, _ = backup_candidates[i]
                used_images.add(img_file)
                image_final_set.append((img_file, lbl_file, image_dir))
                class_final_count[cls] += 1

            for img_file, lbl_file, _ in backup_candidates[extra_originals:]:
                full_img_path = os.path.join(image_dir, img_file)
                if not os.path.exists(full_img_path):
                    continue
                try:
                    img = Image.open(full_img_path).convert("RGB")
                    for j, aug in enumerate(apply_augmentations(img)):
                        if extra_augmented <= 0:
                            break
                        aug_name = f"{os.path.splitext(img_file)[0]}_aug{j}.jpg"
                        aug_lbl = f"{os.path.splitext(img_file)[0]}_aug{j}.txt"
                        aug.save(os.path.join(output_base, "images", split_name, aug_name))
                        shutil.copy2(lbl_file, os.path.join(output_base, "labels", split_name, aug_lbl))
                        used_images.add(aug_name)
                        class_final_count[cls] += 1
                        extra_augmented -= 1
                except Exception as e:
                    print(f"Error augmenting {img_file}: {e}")
                if extra_augmented <= 0:
                    break

        if class_final_count[cls] < target_count:
            needed = target_count - class_final_count[cls]
            fallback_pool = [item for item in all_class_source_images[cls] if item[0] not in used_images]
            random.shuffle(fallback_pool)
            extra_originals = int(0.3 * needed)
            extra_augmented = needed - extra_originals

            for i in range(min(extra_originals, len(fallback_pool))):
                img_file, lbl_file, src_dir = fallback_pool[i]
                used_images.add(img_file)
                image_final_set.append((img_file, lbl_file, src_dir))
                class_final_count[cls] += 1

            for img_file, lbl_file, src_dir in fallback_pool[extra_originals:]:
                full_img_path = os.path.join(src_dir, img_file)
                if not os.path.exists(full_img_path):
                    continue
                try:
                    img = Image.open(full_img_path).convert("RGB")
                    for j, aug in enumerate(apply_augmentations(img)):
                        if extra_augmented <= 0:
                            break
                        aug_name = f"{os.path.splitext(img_file)[0]}_fallback_aug{j}.jpg"
                        aug_lbl = f"{os.path.splitext(img_file)[0]}_fallback_aug{j}.txt"
                        aug.save(os.path.join(output_base, "images", split_name, aug_name))
                        shutil.copy2(lbl_file, os.path.join(output_base, "labels", split_name, aug_lbl))
                        used_images.add(aug_name)
                        class_final_count[cls] += 1
                        extra_augmented -= 1
                except Exception as e:
                    print(f"Error fallback augmenting {img_file}: {e}")
                if extra_augmented <= 0:
                    break

    for img_file, lbl_file, src_dir in image_final_set:
        shutil.copy2(os.path.join(src_dir, img_file), os.path.join(output_base, "images", split_name, img_file))
        shutil.copy2(lbl_file, os.path.join(output_base, "labels", split_name, img_file.replace(".jpg", ".txt")))

    csv_path = os.path.join(output_base, f"class_distribution_{split_name}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class", "Target", "FinalCount"])
        for cls in sorted(class_distribution):
            writer.writerow([cls, class_distribution[cls], class_final_count[cls]])

    plt.figure(figsize=(6, 4))
    plt.bar(class_distribution.keys(), class_final_count.values(), color='skyblue')
    plt.xticks(list(class_distribution.keys()), ["sheep", "person", "wolf", "dog", "horse"])
    plt.ylabel("Number of images")
    plt.title(f"Class distribution in {split_name} set")
    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f"class_distribution_{split_name}.png"))

    print(f"\n✅ {split_name} set created with fallback and global-split rescue.")
    for cls in sorted(class_distribution):
        print(f"Class {cls}: target = {class_distribution[cls]}, final = {class_final_count[cls]}")
    missing_classes = [cls for cls in class_distribution if class_final_count[cls] == 0]
    if missing_classes:
        print(f"⚠️  Warning: Classes missing from {split_name} set: {missing_classes}")


for split in ["train", "val", "test"]:
    process_split(split)

# Global pie chart
full_counts = defaultdict(int)
for split_name in ["train", "val", "test"]:
    csv_path = os.path.join(output_base, f"class_distribution_{split_name}.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = int(row["Class"])
            count = int(row["FinalCount"])
            full_counts[cls] += count

labels = ["person", "sheep", "wolf", "horse", "dog"]
sizes = [full_counts[c] for c in range(5)]
colors = ["#377BC0", "#2C547C", "#6699CC", "#99CCFF", "#CCE5FF"]
explode = [0.1 if i == 0 else 0.2 if i == 1 else 0 for i in range(5)]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
plt.title("Overall Class Distribution", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_base, "overall_class_distribution_pie.png"))