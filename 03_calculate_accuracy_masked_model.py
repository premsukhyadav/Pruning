import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
import urllib.request
import csv
from collections import defaultdict
from PIL import Image

# ----------------------------
# GLOBAL CONFIG
# ----------------------------
CONFIG = {
    "data_dir": "data/original",               # dataset root
    "model_path": "models/masked/resnet18_masked.pth",
    "batch_size": 8,
    "results_dir": "results",
    "num_folders": 20,                         # max number of classes/folders to process (0 = all)
    "imagenet_labels_path": "imagenet_classes.txt"
}

# Create results directory
os.makedirs(CONFIG["results_dir"], exist_ok=True)
detailed_csv = os.path.join(CONFIG["results_dir"], "masked_detailed_report.csv")
summary_csv = os.path.join(CONFIG["results_dir"], "masked_summary_report.csv")

# ----------------------------
# 1. Load Model
# ----------------------------
model = models.resnet18()
if not os.path.exists(CONFIG["model_path"]):
    raise FileNotFoundError(f"Model file '{CONFIG['model_path']}' not found!")

state_dict = torch.load(CONFIG["model_path"], map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Loaded pretrained masked model from {CONFIG['model_path']}")

# ----------------------------
# 2. Preprocessing
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# 3. Dataset (preserve folder order)
# ----------------------------
if not os.path.exists(CONFIG["data_dir"]):
    raise FileNotFoundError(f"Data directory '{CONFIG['data_dir']}' not found!")

# Get folder order from disk
all_folders = [f.name for f in os.scandir(CONFIG["data_dir"]) if f.is_dir()]
if CONFIG["num_folders"] > 0:
    all_folders = all_folders[:CONFIG["num_folders"]]

# Map folder names to class indices
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_folders)}

# Build samples list manually
samples = []
for cls_name in all_folders:
    cls_idx = class_to_idx[cls_name]
    cls_folder = os.path.join(CONFIG["data_dir"], cls_name)
    for fname in os.listdir(cls_folder):
        path = os.path.join(cls_folder, fname)
        if os.path.isfile(path) and fname.lower().endswith((".png", ".jpg", ".jpeg")):
            samples.append((path, cls_idx))

# Create a minimal ImageFolder-like dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None, classes=None):
        self.samples = samples
        self.transform = transform
        self.classes = classes
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

dataset = CustomDataset(samples, transform=preprocess, classes=all_folders)
data_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
print(f"📂 Loaded dataset with {len(dataset)} images, Classes: {dataset.classes}")

# ----------------------------
# 4. ImageNet Class Labels
# ----------------------------
if not os.path.exists(CONFIG["imagenet_labels_path"]):
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(url, CONFIG["imagenet_labels_path"])

with open(CONFIG["imagenet_labels_path"]) as f:
    imagenet_labels = [line.strip() for line in f.readlines()]

# ----------------------------
# 5. Inference + Detailed Report
# ----------------------------
class_total = defaultdict(int)
class_correct = defaultdict(int)
total_images = 0
top1_correct = 0
top5_correct = 0

with open(detailed_csv, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    header = [
        "image_path", "true_label",
        "top1_class", "top1_prob",
        "top2_class", "top2_prob",
        "top3_class", "top3_prob",
        "top4_class", "top4_prob",
        "top5_class", "top5_prob",
    ]
    writer.writerow(header)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            outputs = model(inputs)
            probs, top5_indices = torch.topk(torch.nn.functional.softmax(outputs, dim=1), k=5, dim=1)

            for i in range(inputs.size(0)):
                img_index = batch_idx * CONFIG["batch_size"] + i
                if img_index >= len(dataset.samples):
                    continue

                img_path, _ = dataset.samples[img_index]
                true_label = dataset.classes[labels[i]]

                top_classes = [imagenet_labels[top5_indices[i][j].item()] for j in range(5)]
                top_probs = [probs[i][j].item() * 100 for j in range(5)]

                # write to detailed CSV
                row = [img_path, true_label]
                for cls, prob in zip(top_classes, top_probs):
                    row.extend([cls, f"{prob:.2f}%"])
                writer.writerow(row)

                # update metrics
                class_total[true_label] += 1
                total_images += 1

                if top_classes[0] == true_label:
                    class_correct[true_label] += 1
                    top1_correct += 1

                if true_label in top_classes:
                    top5_correct += 1

print(f"✅ Detailed report saved to {detailed_csv}")

# ----------------------------
# 6. Summary Report
# ----------------------------
with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class_name", "correct", "total", "accuracy (%)"])

    for cls in dataset.classes:  # preserve folder order
        total = class_total[cls]
        correct = class_correct[cls]
        acc = (correct / total * 100) if total > 0 else 0.0
        writer.writerow([cls, correct, total, f"{acc:.2f}"])

    writer.writerow([])
    writer.writerow(["Overall Top-1 Accuracy", top1_correct, total_images, f"{100 * top1_correct / total_images:.2f}"])
    writer.writerow(["Overall Top-5 Accuracy", top5_correct, total_images, f"{100 * top5_correct / total_images:.2f}"])

print(f"✅ Summary report saved to {summary_csv}")
print("\n🎯 Inference + reporting complete.")
