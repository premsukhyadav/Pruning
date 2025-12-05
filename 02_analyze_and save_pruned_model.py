import os
import torch
from torchvision import models

from pruning_methods.analyze_conv_filter import analyze_conv_filter

# ----------------------------
# Helper to fetch layer by string path
# ----------------------------
def get_layer(model, layer_path):
    parts = layer_path.split(".")
    layer = model
    for p in parts:
        if p.isdigit():
            layer = layer[int(p)]
        else:
            layer = getattr(layer, p)
    return layer

# ----------------------------
# Configuration
# ----------------------------
model_path = "models/original/resnet18_state.pth"
masked_model_path = "models/masked/resnet18_masked.pth"

# ----------------------------
# Load pretrained model
# ----------------------------
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

model = models.resnet18()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
print("✅ Loaded pretrained model")

# ----------------------------
# Conv layers to analyze (NO lambda)
# ----------------------------
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

# ---------------------------------------------
#  Manual drop lists for each conv layer
# ---------------------------------------------
MANUAL_DROP = {
    "conv1": [2, 4, 7, 9, 13, 16, 33, 37, 38, 45, 48, 61],

    "layer1.0.conv1": [10, 44, 62],
    "layer1.0.conv2": [18, 24],
    "layer1.1.conv1": [20, 22, 30],
    "layer1.1.conv2": [61],

    "layer2.0.conv1": [62],
    "layer2.0.conv2": [3, 126],
    "layer2.1.conv1": [26, 91],
    "layer2.1.conv2": [57, 65, 121],

    "layer3.0.conv1": [40, 240],
    "layer3.0.conv2": [18],
    "layer3.1.conv1": [148, 155],
    "layer3.1.conv2": [111],

    "layer4.0.conv1": [145, 224],
    "layer4.0.conv2": [73, 488],
    "layer4.1.conv1": [63, 247,],
    "layer4.1.conv2": [90, 105, 336, 471],
}


# ----------------------------
# Filter analysis + pruning
# ----------------------------
for layer_path in CONV_LAYERS:

    conv_layer = get_layer(model, layer_path)
    conv_weights = conv_layer.weight.data

    print(f"\n▶️ Analyzing {layer_path} | shape = {conv_weights.shape}")

    # auto-create folder
    folder_path = f"temp/{layer_path}/"
    plot = not os.path.exists(folder_path)

    # run your filter analysis algorithm
    # drop_list = analyze_conv_filter(conv_weights, conv_name=layer_path, plot=plot)

    #ignore analyze_conv_filter() and use manual list
    drop_list = MANUAL_DROP.get(layer_path, [])
    print(f"⏹️ {layer_path} drop list: {drop_list}")

    # apply pruning mask
    conv_weights[drop_list, :, :, :] = 0.0

# ----------------------------
# Save masked model
# ----------------------------
os.makedirs(os.path.dirname(masked_model_path), exist_ok=True)
torch.save(model.state_dict(), masked_model_path)
print(f"\n✅ Masked model saved to {masked_model_path}")
