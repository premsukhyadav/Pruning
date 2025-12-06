import os
import shutil

# -------------------------------
# User paths (EDIT THESE)
# -------------------------------
classes_file = "../imagenet_classes.txt"
source_root  = "C:/Users/prems/Downloads/imagenet-mini/train"    # contains class folders
target_root  = "../data/original"           # will contain named folders

# -------------------------------
# Load class names
# -------------------------------
with open(classes_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"Loaded {len(class_names)} class names")

# -------------------------------
# Load folder names (random)
# -------------------------------
folder_list = sorted(os.listdir(source_root))
print(f"Found {len(folder_list)} folders")

if len(folder_list) != len(class_names):
    raise ValueError(
        f"Mismatch: {len(folder_list)} folders vs {len(class_names)} class names!"
    )

# -------------------------------
# Create structure and map 1-to-1 (COPY)
# -------------------------------
os.makedirs(target_root, exist_ok=True)

for i, class_name in enumerate(class_names):

    src = os.path.join(source_root, folder_list[i])
    dst = os.path.join(target_root, class_name)

    print(f"[{i}] Copying '{folder_list[i]}' → '{class_name}'")

    # create target folder
    os.makedirs(dst, exist_ok=True)

    # Copy folder contents instead of moving
    shutil.copytree(src, dst, dirs_exist_ok=True)

print("\n✅ Completed class–folder copying!")
