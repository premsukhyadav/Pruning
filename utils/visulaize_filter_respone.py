import os
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# -------------------------------------------------------
# Use your helper to fetch nested layers
# -------------------------------------------------------
def get_layer(model, layer_path):
    parts = layer_path.split(".")
    layer = model
    for p in parts:
        if p.isdigit():
            layer = layer[int(p)]
        else:
            layer = getattr(layer, p)
    return layer


# -------------------------------------------------------
# Load + preprocess image
# -------------------------------------------------------
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)   # [1,3,224,224]


# -------------------------------------------------------
# Save a single feature map (1xHxW)
# -------------------------------------------------------
def save_feature_map(tensor, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)

    fmap = tensor.clone().detach()
    fmap = fmap - fmap.min()
    fmap = fmap / (fmap.max() + 1e-8)

    img = transforms.ToPILImage()(fmap)
    img.save(os.path.join(save_dir, file_name))


# -------------------------------------------------------
# Dump all filters output from a conv layer
# -------------------------------------------------------
def dump_conv_output(input_tensor, conv_layer, layer_name):
    with torch.no_grad():
        output = conv_layer(input_tensor)     # [1, outC, H, W]

    outC = output.shape[1]
    save_dir = f"feature_maps/{layer_name}"
    print(f"[+] Saving {outC} feature maps for layer {layer_name}")

    for i in range(outC):
        fmap = output[0, i].unsqueeze(0)      # [1,H,W]
        save_feature_map(fmap, save_dir, f"{layer_name}_filter{i}.png")

    return output   # feed to next conv layer


# -------------------------------------------------------
# MAIN LOGIC: iterate through selected conv layers
# -------------------------------------------------------
def run_feature_map_visualization(model_path, image_path, conv_layers):
    print("Loading model...")
    model = models.resnet18()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # load first input image
    x = load_image(image_path)

    # process conv-by-conv (not forward pass of entire model)
    for layer_name in conv_layers:
        print(f"\n🔷 Processing layer: {layer_name}")
        conv_layer = get_layer(model, layer_name)

        if not isinstance(conv_layer, torch.nn.Conv2d):
            print(f"Skipping {layer_name}: not Conv2d")
            continue

        x = dump_conv_output(x, conv_layer, layer_name)


# -------------------------------------------------------
# Run script standalone
# -------------------------------------------------------
if __name__ == "__main__":
    MODEL_PATH = "../models/original/resnet18_state.pth"
    SAMPLE_IMAGE = "../data/Chessboard.png"
    CONV_LAYERS = [
        "conv1",

        "layer1.0.conv1", "layer1.0.conv2",
        "layer1.1.conv1", "layer1.1.conv2",

        "layer2.0.conv1", "layer2.0.conv2",
        "layer2.1.conv1", "layer2.1.conv2",

        "layer3.0.conv1", "layer3.0.conv2",
        "layer3.1.conv1", "layer3.1.conv2",

        "layer4.0.conv1", "layer4.0.conv2",
        "layer4.1.conv1", "layer4.1.conv2",
    ]

    run_feature_map_visualization(MODEL_PATH, SAMPLE_IMAGE, CONV_LAYERS)
