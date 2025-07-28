import os
import shutil
import random
import uuid
from pathlib import Path
from collections import defaultdict

from PIL import Image
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Class: YOLOAugmentor
# Description: Handles image augmentations using Albumentations.
# -------------------------------
class YOLOAugmentor:
    def __init__(self):
        # Compose a set of diverse augmentations to increase dataset variability
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),                 # Flip horizontally with 50% chance
            A.RandomBrightnessContrast(p=0.5),       # Randomly adjust brightness & contrast
            A.RGBShift(p=0.2),                        # Random shift of RGB channels
            A.RandomShadow(p=0.3),                    # Add random shadows for realism
            A.Blur(p=0.2),                            # Gaussian blur effect
            A.MotionBlur(p=0.2),                      # Motion blur simulating camera movement
            A.Rotate(limit=15, p=0.3),                # Rotate image up to ±15 degrees
            A.RandomCrop(width=640, height=640, p=0.3) # Crop images to 640x640 for YOLO input
        ])

    def augment(self, img_path):
        """
        Load an image from disk, ensure it's large enough for augmentation crop,
        apply the augmentation pipeline, and return the augmented image.
        
        Args:
            img_path (Path or str): Path to input image file.
            
        Returns:
            np.ndarray: Augmented image as a numpy array (RGB).
        """
        crop_h, crop_w = 640, 640  # your crop size
        
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Resize if image is smaller than crop size
        if h < crop_h or w < crop_w:
            scale = max(crop_h / h, crop_w / w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img_np = np.array(img)
        
        augmented = self.transform(image=img_np)
        return augmented['image']

# -------------------------------
# Class: SplitStatsTracker
# Description: Keeps track of per-class sample counts and video diversity within a dataset split.
# -------------------------------
class SplitStatsTracker:
    def __init__(self, classes):
        """
        Initialize data structures for tracking.

        Args:
            classes (iterable): List or set of class IDs being tracked.
        """
        self.class_counts = {cls: 0 for cls in classes}      # Mapping: class_id -> total samples count
        self.video_ids = defaultdict(set)                    # Mapping: class_id -> set of unique video IDs
        self.samples = []                                    # List to keep (class_id, video_id, filename) tuples

    def add_sample(self, cls, video_id, img_name):
        """
        Record a single sample addition for statistics.
        
        Args:
            cls (int): Class ID of the sample.
            video_id (str): Video identifier extracted from filename.
            img_name (str): Filename of the image.
        """
        self.class_counts[cls] += 1
        self.video_ids[cls].add(video_id)
        self.samples.append((cls, video_id, img_name))

    def to_dataframe(self):
        """
        Export the tracked statistics as a pandas DataFrame for analysis or export.

        Returns:
            pd.DataFrame: Columns ['class', 'count', 'video_diversity'].
        """
        data = []
        for cls, count in self.class_counts.items():
            data.append({
                'class': cls,
                'count': count,
                'video_diversity': len(self.video_ids[cls]),
            })
        return pd.DataFrame(data)

# -------------------------------
# Class: DatasetBuilder
# Description: Builds balanced YOLO datasets with train/val/test splits, applies augmentations,
# and tracks per-class statistics including video diversity. Supports plotting detailed reports.
# -------------------------------
class DatasetBuilder:
    def __init__(self, source_dir, output_dir, total_targets, split_proportions):
        """
        Initialize dataset builder.

        Args:
            source_dir (str or Path): Directory containing images and labels (.txt files).
            output_dir (str or Path): Directory to save output splits and reports.
            total_targets (dict): Mapping class_id -> total number of samples desired (all splits combined).
            split_proportions (dict): Mapping split_name ('train', 'val', 'test') -> proportion (sum=1).
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.total_targets = total_targets
        self.split_proportions = split_proportions
        self.augmentor = YOLOAugmentor()
        self.split_class_images = {}  # To store images per split and class
        self.split_reports = {}  # Will hold pandas DataFrames with stats per split

    def load_image_paths(self, split_name):
        """
        Load images and labels for the given split from dataset/images/{split_name}
        and dataset/labels/{split_name}, grouped by class (first class id in label file).

        Supports multiple image formats.

        Args:
            split_name (str): "train", "val", or "test"

        Returns:
            dict: {class_id: [(img_path, label_path), ...]}
        """
        img_dir = self.source_dir / "images" / split_name
        lbl_dir = self.source_dir / "labels" / split_name

        # Supported image extensions (lowercase)
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        # Gather all image files with supported extensions (case-insensitive)
        img_paths = []
        for ext in img_extensions:
            img_paths.extend(img_dir.glob(f"*{ext}"))
            img_paths.extend(img_dir.glob(f"*{ext.upper()}"))  # Also include uppercase extensions

        by_class = defaultdict(list)

        for img_path in img_paths:
            # Build corresponding label path (replace extension with .txt)
            label_path = lbl_dir / (img_path.stem + ".txt")

            if label_path.exists():
                with open(label_path) as f:
                    lines = f.readlines()
                    if lines:
                        class_id = int(lines[0].split()[0])
                        by_class[class_id].append((img_path, label_path))

        return by_class


    def compute_split_targets(self):
        """
        Calculate per-split per-class target counts by applying proportions.

        Returns:
            dict: split_name -> dict of class_id -> int (target count for that split).
        """
        split_targets = {split: {} for split in self.split_proportions}
        for cls, total in self.total_targets.items():
            for split, prop in self.split_proportions.items():
                split_targets[split][cls] = int(round(prop * total))
        return split_targets
    
    @staticmethod
    def sort_by_video_diversity(samples):
        """
        Interleave samples from different videos to maximize diversity.
        Assumes video_id is prefix before first underscore in filename.
        """
        video_groups = defaultdict(list)
        for img_path, lbl_path in samples:
            video_id = img_path.stem.split("_")[0]
            video_groups[video_id].append((img_path, lbl_path))
        interleaved = []
        while video_groups:
            for vid in list(video_groups.keys()):
                if video_groups[vid]:
                    interleaved.append(video_groups[vid].pop())
                else:
                    del video_groups[vid]
        return interleaved

    def borrow_between_val_test(self, class_id, target_split, needed, exclude_files):
        """
        Borrow up to 50% of needed samples for a given class into target_split.
        Only borrowing between 'val' and 'test' is allowed.

        Args:
            class_id (int): The class needing more samples.
            target_split (str): Either 'val' or 'test'.
            needed (int): Number of samples still missing.
            exclude_files (set): Image filenames already used.

        Returns:
            list of (img_path, lbl_path): Borrowed image-label pairs.
        """
        if target_split == "train":
            print(f"[BLOCKED] Borrowing into 'train' split is not allowed for class {class_id}.")
            return []

        if needed <= 0:
            print(f"[INFO] No borrowing needed for class {class_id} in split '{target_split}'.")
            return []

        source_split = "val" if target_split == "test" else "test"
        source_samples = self.split_class_images.get(source_split, {}).get(class_id, [])

        if not source_samples:
            print(f"[BORROW] No samples found in '{source_split}' to borrow for class {class_id}.")
            return []

        # Filter out already used files
        available = [s for s in source_samples if s[0].name not in exclude_files]

        if not available:
            print(f"[BORROW] No unused samples available to borrow from '{source_split}' for class {class_id}.")
            return []

        # Optional: sort available by video diversity or other criteria here

        borrow_count = min(len(available), (needed + 1) // 2)
        borrowed = available[:borrow_count]

        print(f"[BORROW] Borrowing {len(borrowed)}/{needed} samples for class {class_id} from '{source_split}' → '{target_split}'.")
        return borrowed
        
    def build_split(self, split_name, class_targets):
        """
        Build a dataset split (train/val/test) with balanced class distributions.

        Args:
            split_name (str): Name of the split ('train', 'val', or 'test').
            class_targets (dict): Target number of samples per class for this split.

        Process:
        - For each class, attempt to copy available original samples.
        - If insufficient, borrow between 'val' and 'test' only (never from or into 'train').
        - If still insufficient, apply augmentations using previously copied images.
        - Tracks and saves statistics per class.
        """
        print(f"[INFO] Building split: {split_name}")
        class_data = self.split_class_images.get(split_name, {})

        # Prepare output directories
        split_dir = self.output_dir 
        img_dir = split_dir / "images" / split_name
        lbl_dir = split_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        stats = SplitStatsTracker(class_targets.keys())
        used_files = set()  # Track filenames already used in this split

        for cls, target_count in class_targets.items():
            print(f"\n[CLASS] Processing class '{cls}' → target = {target_count} images")

            samples = class_data.get(cls, [])

            # Step 1: Copy original samples
            count = 0
            i = 0
            while count < target_count and i < len(samples):
                img_path, lbl_path = samples[i]
                i += 1

                if img_path.name in used_files:
                    continue  # Avoid duplicates

                shutil.copy(img_path, img_dir / img_path.name)
                shutil.copy(lbl_path, lbl_dir / lbl_path.name)

                video_id = img_path.stem.split("_")[0]
                stats.add_sample(cls, video_id, img_path.name)
                used_files.add(img_path.name)
                count += 1
                print(f"[COPY] Copied {img_path.name}")

            # Step 2: Borrow samples if still missing and split is 'val' or 'test'
            if count < target_count and split_name in ("val", "test"):
                needed = target_count - count
                borrowed = self.borrow_between_val_test(cls, split_name, needed, used_files)
                for img_path, lbl_path in borrowed:
                    if img_path.name in used_files:
                        continue

                    shutil.copy(img_path, img_dir / img_path.name)
                    shutil.copy(lbl_path, lbl_dir / lbl_path.name)

                    video_id = img_path.stem.split("_")[0]
                    stats.add_sample(cls, video_id, img_path.name)
                    used_files.add(img_path.name)
                    count += 1
                    print(f"[BORROW] Borrowed {img_path.name}")

            # Step 3: If still missing samples, perform augmentation
            if count < target_count:
                print(f"[AUGMENT] Not enough originals for class '{cls}'. Applying augmentation...")

                candidate_files = sorted(used_files)
                if not candidate_files:
                    print(f"[WARNING] No base images available for augmentation for class '{cls}' in split '{split_name}'")
                    continue

                aug_index = 0
                while count < target_count:
                    base_img_name = candidate_files[aug_index % len(candidate_files)]
                    base_img_path = img_dir / base_img_name
                    base_lbl_path = lbl_dir / Path(base_img_name).with_suffix('.txt')

                    if not base_lbl_path.exists():
                        print(f"[WARNING] Label missing for {base_img_name}. Skipping augmentation.")
                        aug_index += 1
                        continue

                    aug_img = self.augmentor.augment(base_img_path)
                    uid = uuid.uuid4().hex[:8]
                    new_img_name = f"{cls}_aug_{uid}.jpg"
                    new_lbl_name = f"{cls}_aug_{uid}.txt"

                    Image.fromarray(aug_img).save(img_dir / new_img_name)
                    shutil.copy(base_lbl_path, lbl_dir / new_lbl_name)

                    video_id = new_img_name.split("_")[0]
                    stats.add_sample(cls, video_id, new_img_name)
                    used_files.add(new_img_name)
                    count += 1

                    print(f"[AUGMENT] Created augmented image {new_img_name} for class '{cls}'")
                    aug_index += 1

            if count < target_count:
                print(f"[WARNING] Only {count}/{target_count} samples generated for class '{cls}' in split '{split_name}'")
            else:
                print(f"[DONE] Finished class '{cls}' with {count} images.")

        # Save stats to CSV
        df = stats.to_dataframe()
        df.to_csv(split_dir / f"{split_name}_distribution.csv", index=False)
        self.split_reports[split_name] = df

        total_samples = sum(stats.class_counts.values())
        print(f"[INFO] Split '{split_name}' dataset created with total {total_samples} samples.\n")
        return df

    def build_all_splits(self):
        """
        Build train, val, test splits sequentially based on target distribution and proportions.
        """
        # Step 1 : We start by loading all images per split
        for split in ['train', 'val', 'test']:
            self.split_class_images[split] = self.load_image_paths(split)

        # Step 2 : We build the splits
        split_targets = self.compute_split_targets()
        for split_name, class_targets in split_targets.items():
            self.build_split(split_name, class_targets)

    def generate_distribution_report(self):
        """
        Combine all split reports into a single DataFrame with multi-index split + class.

        Returns:
            pd.DataFrame: Combined DataFrame with columns ['split', 'class', 'count', 'video_diversity'].
        """
        combined = pd.concat(self.split_reports.values(), keys=self.split_reports.keys(), names=["split"])
        return combined.reset_index()

    @staticmethod
    def plot_distribution_pies_and_diversity(report_df, output_dir, class_map=None):
        """
        Generate high-quality pie charts (class distribution) and bar charts (video diversity)
        for each split and overall.

        Args:
            report_df (pd.DataFrame): DataFrame with columns ['split', 'class', 'count', 'video_diversity'].
            output_dir (str or Path): Directory to save plots.
            class_map (dict, optional): Mapping class_id -> human-readable class name.
        """
        output_dir = Path(output_dir)
        plots_dir = output_dir / "distribution_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        colors = plt.get_cmap("tab20").colors
        splits = report_df['split'].unique()

        def label_name(cls_id):
            if class_map and cls_id in class_map:
                return class_map[cls_id]
            return str(cls_id)

        for split in splits:
            df_split = report_df[report_df['split'] == split]

            # Pie chart for class distribution
            plt.figure(figsize=(9, 9), dpi=150)
            sizes = df_split['count']
            labels = [label_name(c) for c in df_split['class']]
            plt.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=140,
                colors=colors[:len(sizes)],
                shadow=True,
                textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'navy'}
            )
            plt.title(f"Class Distribution in '{split}' Split", fontsize=20, fontweight='bold', color='darkred')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(plots_dir / f"{split}_class_distribution_pie.png")
            plt.close()

            # Bar chart for video diversity
            plt.figure(figsize=(10, 6), dpi=150)
            classes = df_split['class']
            diversity = df_split['video_diversity']
            names = [label_name(c) for c in classes]

            bars = plt.bar(names, diversity, color=colors[:len(classes)])
            plt.xlabel("Class", fontsize=14, fontweight='bold')
            plt.ylabel("Number of Unique Videos", fontsize=14, fontweight='bold')
            plt.title(f"Video Diversity per Class in '{split}' Split", fontsize=18, fontweight='bold', color='darkgreen')
            plt.xticks(rotation=45, fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.savefig(plots_dir / f"{split}_video_diversity_bar.png")
            plt.close()

        # Overall summary plots
        overall = report_df.groupby('class').sum().reset_index()
        labels = [label_name(c) for c in overall['class']]

        # Overall pie chart
        plt.figure(figsize=(9, 9), dpi=150)
        plt.pie(
            overall['count'],
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors[:len(overall)],
            shadow=True,
            textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'navy'}
        )
        plt.title("Overall Class Distribution", fontsize=20, fontweight='bold', color='darkred')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(plots_dir / "overall_class_distribution_pie.png")
        plt.close()

        # Overall video diversity bar chart
        plt.figure(figsize=(10, 6), dpi=150)
        bars = plt.bar(labels, overall['video_diversity'], color=colors[:len(overall)])
        plt.xlabel("Class", fontsize=14, fontweight='bold')
        plt.ylabel("Number of Unique Videos", fontsize=14, fontweight='bold')
        plt.title("Overall Video Diversity per Class", fontsize=18, fontweight='bold', color='darkgreen')
        plt.xticks(rotation=45, fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(plots_dir / "overall_video_diversity_bar.png")
        plt.close()


# -----------------------------------------
# Example usage
# -----------------------------------------
if __name__ == "__main__":
    # Define total target samples per class (all splits combined)
    total_target_distribution = {
        0: 10000,   # person
        18: 12000,  # sheep
        19: 6000,   # cow
        16: 7500,   # dog
        21: 7000,   # bear
        80: 9000,   # wolf
        81: 6500,   # coyote
        82: 6500,   # fox
        83: 9000,   # redfox
        84: 6000    # wild dog
    }
    total = 98167

    # Define split proportion (train + val + test = 1)
    # Split proportions 
    split_proportions = {
        "train": 58900 / 98167,  # ~0.60
        "val":   19633 / 98167,  # ~0.20
        "test":  19633 / 98167   # ~0.20
    }

    split_proportions = {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2
    }
    


    # Optional class ID → name mapping for plot labels
    class_map = {
        0: "person",
        18: "sheep",
        19: "cow",
        16: "dog",
        21: "bear",
        80: "wolf",
        81: "coyote",
        82: "fox",
        83: "redfox",
        84: "wild dog",
    }
    
    source_dir = "dataset_plus_"  # 
    output_dir = "dataset"  # Output folder
    #source_dir = "/home/pruebaaitor/Desktop/S_YOLOv8/newExperiment/remapped_dset"  # 
    #output_dir = "dataset_pred"  # Output folder

     # Initialize DatasetBuilder with source and output directories
    builder = DatasetBuilder(
        source_dir=source_dir,
        output_dir=output_dir,
        total_targets=total_target_distribution,
        split_proportions=split_proportions
    )
    
    # Build train, val, test splits according to targets and proportions
    # with borrowing and augmentation as needed
    builder.build_all_splits()

    # Generate and save combined distribution report

    # Generate combined report DataFrame
    report_df = builder.generate_distribution_report()

    report_df.to_csv(output_dir + "/split_distribution_report.csv", index=False)
    print(report_df.head())

    # Generate detailed plots for class distribution and video diversity
    builder.plot_distribution_pies_and_diversity(report_df, output_dir= output_dir, class_map=class_map)


