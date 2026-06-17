import os

os.environ["MXNET_SUBGRAPH_VERBOSE"] = "0"
os.environ["MKLDNN_VERBOSE"] = "0"

import argparse
import cv2
import numpy as np
from tqdm import tqdm
from evaluations.FaceImageQuality.face_image_quality import SER_FIQ

def parse_args():
    parser = argparse.ArgumentParser(description='SER-FIQ Score Calculator')

    # 默认路径指向你之前脚本生成的输出目录
    parser.add_argument('--base_dir', default='evaluation_outputs',
                        help='包含 clean_swaps 和 protected_swaps 的基础目录')

    # 也可以单独指定某个文件夹跑
    parser.add_argument('--target_folder', type=str, default=None,
                        help='如果只跑单个文件夹，请指定此参数 (例如: /path/to/images)')

    parser.add_argument('--gpu', type=int, default=0, help='CUDA device id (设为 -1 使用 CPU)')
    parser.add_argument('--T', type=int, default=100, help='Dropout 采样次数 (标准为100，测试可调小)')

    args = parser.parse_args()
    return args


def calculate_folder_score(folder_path, ser_fiq_model, T=100):
    """计算单个文件夹的平均 SER-FIQ 分数"""
    if not os.path.exists(folder_path):
        print(f"[错误] 文件夹不存在: {folder_path}")
        return 0.0

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"[警告] 文件夹为空: {folder_path}")
        return 0.0

    scores = []
    print(f"正在处理文件夹: {os.path.basename(folder_path)} (共 {len(image_files)} 张)")

    for img_name in tqdm(image_files, desc="Calculating SER-FIQ"):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # SER-FIQ 需要对齐
        try:
            aligned_img = ser_fiq_model.apply_mtcnn(img)
            if aligned_img is not None:
                # 获取分数
                score = ser_fiq_model.get_score(aligned_img, T=T)
                scores.append(score)
        except Exception as e:
            # 捕获可能的 MXNet 错误，防止单张图报错中断整个进程
            pass

    avg_score = np.mean(scores) if scores else 0.0
    return avg_score


def main():
    args = parse_args()

    # 初始化模型
    gpu_id = args.gpu if args.gpu >= 0 else None
    print(f"正在加载 SER-FIQ 模型 (GPU: {gpu_id})...")
    ser_fiq = SER_FIQ(gpu=gpu_id)

    # 模式 1: 指定了单个目标文件夹
    if args.target_folder:
        score = calculate_folder_score(args.target_folder, ser_fiq, args.T)
        print(f"\n文件夹: {args.target_folder}")
        print(f"平均 SER-FIQ 分数: {score:.4f}")

    # 模式 2: 默认模式，自动跑 evaluation_outputs 下的两个子文件夹
    else:
        clean_dir = os.path.join(args.base_dir, "clean_swaps")
        protected_dir = os.path.join(args.base_dir, "protected_swaps")

        print("\n--- 开始评估 Clean Swaps ---")
        clean_score = calculate_folder_score(clean_dir, ser_fiq, args.T)

        print("\n--- 开始评估 Protected Swaps ---")
        protected_score = calculate_folder_score(protected_dir, ser_fiq, args.T)

        print("\n" + "=" * 40)
        print("SER-FIQ 最终评估结果")
        print("=" * 40)
        print(f"Clean Swaps (基准质量):     {clean_score:.4f}")
        print(f"Protected Swaps (防御后质量): {protected_score:.4f}")
        print("-" * 40)
        if clean_score > 0:
            drop_rate = (clean_score - protected_score) / clean_score * 100
            print(f"质量下降率: {drop_rate:.2f}% (越高代表防御造成的画质破坏越大)")
        print("=" * 40)


if __name__ == '__main__':
    main()