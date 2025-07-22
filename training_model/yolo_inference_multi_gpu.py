import time
import torch
import cv2
import gc
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import random
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName

# -------------------------------
# GPU detection using NVIDIA NVML
# -------------------------------
def get_available_gpus():
    """
    Initialize NVML and return list of available GPUs with their indices and names.
    """
    nvmlInit()
    gpu_list = []
    for idx in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(idx)
        raw_name = nvmlDeviceGetName(handle)
        gpu_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name
        gpu_list.append((idx, gpu_name))
    return gpu_list

# -------------------------------
# Load images from a directory
# -------------------------------
def load_images(image_dir, max_images=5):
    """
    Load up to `max_images` from the specified directory.

    Parameters:
    - image_dir (str): Path to the image directory.
    - max_images (int): Maximum number of images to load.

    Returns:
    - list of loaded images (as numpy arrays).
    """
    image_paths = list(Path(image_dir).glob("*.jpg"))
    selected = random.sample(image_paths, min(max_images, len(image_paths)))
    return [cv2.imread(str(p)) for p in selected]

# -------------------------------
# Utility: model size and parameters
# -------------------------------
def get_model_size(model_path):
    """
    Get the size of a model file in megabytes.
    """
    return round(os.path.getsize(model_path) / (1024 * 1024), 2)

def count_model_params(model):
    """
    Count the total number of learnable parameters in the model.
    """
    return sum(p.numel() for p in model.model.parameters())

# -------------------------------
# Compute validation metrics
# -------------------------------
def get_model_metrics_global(metrics):
    precision = sum(metrics.p) / len(metrics.p) if metrics.p else None
    recall = sum(metrics.r) / len(metrics.r) if metrics.r else None
    f1 = sum(metrics.f1) / len(metrics.f1) if metrics.f1 else None
    map50 = metrics.map50 if hasattr(metrics, 'map50') else None

    return {
        "mAP@0.5": round(map50, 3) if map50 is not None else None,
        "Precision": round(precision, 3) if precision is not None else None,
        "Recall": round(recall, 3) if recall is not None else None,
        "F1 score": round(f1, 3) if f1 is not None else None
    }

def get_model_metrics_global2(metrics):
    precision = metrics.p[-1] if metrics.p else None
    recall = metrics.r[-1] if metrics.r else None
    f1 = metrics.f1[-1] if metrics.f1 else None
    map50 = metrics.map50 if hasattr(metrics, 'map50') else None

    return {
        "mAP@0.5": round(map50, 3) if map50 is not None else None,
        "Precision": round(precision, 3) if precision is not None else None,
        "Recall": round(recall, 3) if recall is not None else None,
        "F1 score": round(f1, 3) if f1 is not None else None
    }

def get_model_metrics(model, device, split="val"):
    """
    Run validation and extract detection metrics (mAP@0.5, precision, recall, F1).

    Parameters:
    - model (YOLO): Ultralytics YOLO model.
    - device (str): GPU device string (e.g., "cuda:0").

    Returns:
    - dict with computed metrics.
    """
    try:
        val_results = model.val(device=device, split=split)
        metrics = getattr(val_results, 'box', None)
        if metrics is None:
            print("Warning: 'box' attribute not found in validation results")
            return {"mAP@0.5": None, "Precision": None, "Recall": None, "F1 score": None}

        # récupération scalaire globale
        precision = float(metrics.p[-1]) if metrics.p is not None else None
        recall = float(metrics.r[-1]) if metrics.r is not None else None
        f1 = float(metrics.f1[-1]) if metrics.f1 is not None else None
        map50 = float(metrics.map50) if hasattr(metrics, 'map50') else None

        # calcul F1 si nécessaire (optionnel, déjà dans metrics.f1[-1])
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1_calc = 2 * precision * recall / (precision + recall)
        else:
            f1_calc = None

        return {
            "mAP@0.5": round(map50, 3) if map50 is not None else None,
            "Precision": round(precision, 3) if precision is not None else None,
            "Recall": round(recall, 3) if recall is not None else None,
            "F1 score": round(f1_calc, 3) if f1_calc is not None else None
        }
    except Exception as e:
        print(f"Warning: Could not retrieve metrics due to error: {e}")
        return {"mAP@0.5": None, "Precision": None, "Recall": None, "F1 score": None}

# -------------------------------
# Inference benchmarking
# -------------------------------
def evaluate_model(name, model_path, images, gpu_idx, gpu_name):
    """
    Benchmark inference speed and accuracy metrics of a YOLO model on a specific GPU.

    Parameters:
    - name (str): Model name.
    - model_path (str): Path to the .pt model file.
    - images (list): List of input images (numpy arrays).
    - gpu_idx (int): GPU index.
    - gpu_name (str): GPU label for results.

    Returns:
    - dict with performance metrics and inference results.
    """
    device_str = f"cuda:{gpu_idx}"
    model = None  # Initialize variable
    try:
        model = YOLO(model_path)
        model_size = get_model_size(model_path)
        num_params = count_model_params(model)
        #metrics = get_model_metrics(model, device_str)
        metrics = get_model_metrics(model, device=device_str, split="test")

        # Warm-up to stabilize inference time
        _ = model.predict(images[0], device=device_str, verbose=False, project="result/{name}")

        start = time.time()
        for img in images:
            _ = model.predict(img, device=device_str, verbose=False)
        end = time.time()

        duration = end - start
        fps = len(images) / duration
        result = {
            "GPU": gpu_name,
            "Model": name,
            "Images processed": len(images),
            "Total time (s)": round(duration, 2),
            "Average FPS": round(fps, 2),
            "Model size (MB)": model_size,
            "Parameters": num_params,
            "mAP@0.5": metrics["mAP@0.5"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 score": metrics["F1 score"],
            "Error": ""
        }
    except Exception as e:
        result = {
            "GPU": gpu_name,
            "Model": name,
            "Images processed": 0,
            "Total time (s)": 0,
            "Average FPS": 0,
            "Model size (MB)": 0,
            "Parameters": 0,
            "mAP@0.5": None,
            "Precision": None,
            "Recall": None,
            "F1 score": None,
            "Error": str(e)
        }
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return result

# -------------------------------
# Plot results with high resolution
# -------------------------------
def plot_comparison(df):
    """
    Generate high-resolution bar plots for each metric and save as PNG.
    """
    if "Error" not in df.columns:
        print("The 'Error' column does not exist in the DataFrame.")
        return
    
    df_no_errors = df[df["Error"] == ""]

    if df_no_errors.empty:
        print("No valid data without errors to plot.")
        return

    metrics = ["Average FPS", "Model size (MB)", "Parameters", "mAP@0.5", "Precision", "Recall", "F1 score"]

    # Check that all metric columns exist
    missing_cols = [col for col in metrics if col not in df_no_errors.columns]
    if missing_cols:
        print(f"Missing columns in the DataFrame: {missing_cols}")
        # Remove missing columns from metrics to plot the rest
        metrics = [col for col in metrics if col in df_no_errors.columns]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        labels = df_no_errors["Model"] + " (" + df_no_errors["GPU"] + ")"
        bars = plt.bar(labels, df_no_errors[metric])
        plt.title(f"{metric} per Model/GPU")
        plt.ylabel(metric)
        plt.xlabel("Model (GPU)")
        plt.xticks(rotation=45)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"chart_{metric.replace(' ', '_')}.png", dpi=300)
        plt.close()

    # Combined chart (optional)
    try:
        combined_df = df_no_errors.set_index("Model")[metrics]
        if not combined_df.empty:
            combined_df.plot(
                kind='bar', subplots=True, figsize=(14, 16), layout=(4, 2), legend=False, sharex=True,
                title="YOLO Model Comparison Overview"
            )
            plt.tight_layout()
            plt.savefig("model_comparison.png", dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error during combined plot: {e}")

    print("\n📈 All charts saved in high resolution (DPI=300)")
    
# -------------------------------
# Main execution block
# -------------------------------
def main():
    """
    Main procedure for multi-GPU inference benchmarking.
    """
    image_dir = "/home/pruebaaitor/Desktop/S_YOLOv8/dataset_balanced/images/test/"
    #image_dir = f"{data_path}custom_data5.yaml"
    model_path = "/home/pruebaaitor/Desktop/S_YOLOv8/model_run_update/newruns/"
    rest_path = "/train/weights/best.pt"
    
    model_paths = {
        "YOLOv8n": f"{model_path}detect_v8{rest_path}",
        "YOLOv10": f"{model_path}detect_v10{rest_path}",
        "YOLOv11": f"{model_path}detect_v11{rest_path}",
        "YOLOv12": f"{model_path}detect_v12{rest_path}",
        "RTDETR": f"{model_path}detect_rtdetr_{rest_path}",
    }

    model_paths = {
        "SHEPHERD": f"{model_path}detect_v8CNB{rest_path}",
    }



    print("📁 Loading test images...")
    images = load_images(image_dir, max_images=5)

    print("🧠 Detecting available GPUs...")
    gpus = get_available_gpus()
    for idx, name in gpus:
        print(f" - GPU {idx} : {name}")

    results = []
    for gpu_idx, gpu_name in gpus:
        print(f"\n🚀 Running inference on GPU {gpu_idx} ({gpu_name})")
        for name, path in model_paths.items():
            print(f" ▶️ Model: {name}")
            res = evaluate_model(name, path, images, gpu_idx, gpu_name)
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("inference_multi_gpu_results.csv", index=False)

# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    main()