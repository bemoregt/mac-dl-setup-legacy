# resnet50_onnx_vulkan.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time

def main():
    # 1. Vulkan 세션 생성
    print("🔄 Vulkan 세션 초기화 중...")
    session = ort.InferenceSession(
        "resnet50.onnx",
        providers=[
            ('VulkanExecutionProvider', {"device_id": 0}),
            'CPUExecutionProvider' # 폴백
        ]
    )
    print(f"✅ 활성 Provider: {session.get_providers()}")

    # 2. 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open("sample.jpg").convert('RGB')
    input_tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)

    # 3. 벤치마크
    # 웜업
    for _ in range(5):
        _ = session.run(None, {'input': input_tensor})
    
    # 속도 측정
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        output = session.run(None, {'input': input_tensor})
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000
    predicted_idx = np.argmax(output[0])

    print(f"\n✅ ONNX Runtime Vulkan 추론 완료")
    print(f"   평균 추론 시간: {avg_time:.2f} ms")
    print(f"   예측된 클래스 인덱스: {predicted_idx}")

if __name__ == "__main__":
    main()
