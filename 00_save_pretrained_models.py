import torch
import torchvision.models as models
import os
from torchinfo import summary
from torchvision.models import ResNet18_Weights

# ----------------------------
# 1. Load Pretrained Model
# ----------------------------
model_name = "resnet18"
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
weights = ResNet18_Weights.IMAGENET1K_V1
print(weights.transforms())
model.eval()
print(f"Loaded pretrained model: {model_name}")

# ----------------------------
# 2. Save Model Variants
# ----------------------------
save_dir = "models/original"
os.makedirs(save_dir, exist_ok=True)

# a) Save state_dict (weights only)
state_path = os.path.join(save_dir, f"{model_name}_state.pth")
torch.save(model.state_dict(), state_path)
print(f"✅ State dict saved to: {state_path}")

# b) Save full model (architecture + weights)
full_model_path = os.path.join(save_dir, f"{model_name}_full.pt")
torch.save(model, full_model_path)
print(f"✅ Full model saved to: {full_model_path}")

# c) Export to ONNX (for Netron visualization)
onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
dummy_input = torch.randn(1, 3, 224, 224)  # typical ResNet input
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
)
print(f"✅ ONNX model exported to: {onnx_path}")

# ----------------------------
# 3. Show Model Summary
# ----------------------------
print("\n🔹 Model Summary:")
summary(model, input_size=(1, 3, 224, 224),
        col_names=("input_size", "output_size", "num_params", "kernel_size", "trainable"))
