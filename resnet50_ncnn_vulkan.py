import argparse
import time

import ncnn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]


def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    chw = transform(img).unsqueeze(0).numpy().astype(np.float32)  # (1,3,224,224)
    hwc = chw[0].transpose(1, 2, 0).copy()  # (224,224,3)
    return ncnn.Mat(hwc)


def build_net(param, model, use_vulkan):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = use_vulkan
    net.opt.use_fp16_packed = use_vulkan
    net.opt.use_fp16_storage = use_vulkan
    net.opt.use_fp16_arithmetic = False
    net.load_param(param)
    net.load_model(model)
    return net


def infer(net, in_mat, num_runs):
    for _ in range(5):
        ex = net.create_extractor()
        ex.input("in0", in_mat)
        _, out = ex.extract("out0")

    start = time.time()
    for _ in range(num_runs):
        ex = net.create_extractor()
        ex.input("in0", in_mat)
        _, out = ex.extract("out0")
    elapsed = (time.time() - start) / num_runs * 1000

    logits = np.asarray(out).flatten()
    return elapsed, logits


def main():
    parser = argparse.ArgumentParser(description="ncnn Vulkan ResNet50 inference")
    parser.add_argument("--image", default="sample.jpg")
    parser.add_argument("--param", default="resnet50.ncnn.param")
    parser.add_argument("--model", default="resnet50.ncnn.bin")
    parser.add_argument("--labels", default="imagenet_classes.txt")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="compare against CPU backend")
    args = parser.parse_args()

    in_mat = preprocess(args.image)
    labels = load_labels(args.labels)

    print("ncnn:", ncnn.__version__)
    ncnn.create_gpu_instance()
    print("GPU count:", ncnn.get_gpu_count())
    ncnn.destroy_gpu_instance()

    if not args.cpu:
        net = build_net(args.param, args.model, use_vulkan=True)
        ms, logits = infer(net, in_mat, args.runs)
        top5 = np.argsort(logits)[::-1][:5]
        print(f"\n[Vulkan] 평균 추론 시간: {ms:.2f} ms ({args.runs}회)")
        print("[Vulkan] Top-5 예측:")
        for i, idx in enumerate(top5):
            print(f"  {i+1}. {labels[idx]} (class {idx})")
    else:
        gpu_ms, gpu_logits = infer(build_net(args.param, args.model, True), in_mat, args.runs)
        cpu_ms, cpu_logits = infer(build_net(args.param, args.model, False), in_mat, args.runs)
        top5 = np.argsort(gpu_logits)[::-1][:5]
        print(f"\n[Vulkan] 평균: {gpu_ms:.2f} ms | [CPU] 평균: {cpu_ms:.2f} ms")
        print(f"속도 향상: {cpu_ms / gpu_ms:.1f}x")
        print("Top-5 예측:")
        for i, idx in enumerate(top5):
            print(f"  {i+1}. {labels[idx]} (class {idx})")


if __name__ == "__main__":
    main()
