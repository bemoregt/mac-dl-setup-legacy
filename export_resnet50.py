# export_resnet50_onnx.py
import torch
import torchvision.models as models

# 모델 로드 및 ONNX 변환
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "resnet50.onnx",
    export_params=True, opset_version=14,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✅ resnet50.onnx 저장 완료")
