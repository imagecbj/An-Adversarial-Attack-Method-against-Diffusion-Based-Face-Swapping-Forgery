import os
import glob
import numpy as np
import torch
from PIL import Image
import cv2

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
from tqdm import tqdm

# ==============================================================================
# 1. 配置区域
# ==============================================================================
###last results###
RESULTS_BASE_DIR = "/home/lab/workspace/works/cyx/faceshield-main/results/result_2025-12-04-17-09-02"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
def preprocess_image_for_lpips(image_np, device):
    img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    return (img_tensor / 127.5 - 1.0).to(device)


def generate_and_save_residual_map(ori_np, adv_np, save_path):

    # 1. 计算绝对差值
    diff = np.abs(adv_np.astype(np.float32) - ori_np.astype(np.float32))

    # 2. 对通道求平均（或者转为灰度），得到 [H, W] 的强度图
    diff_gray = np.mean(diff, axis=2).astype(np.uint8)

    # 3. 放大差异以便肉眼观察 (归一化到 0-255)
    if np.max(diff_gray) > 0:
        normalized_diff = cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    else:
        normalized_diff = diff_gray
    # 4. 应用伪彩色映射
    heatmap = cv2.applyColorMap(normalized_diff, cv2.COLORMAP_JET)

    # 5. 保存图像
    cv2.imwrite(save_path, heatmap)


# ==============================================================================
# 3. 主评估流程
# ==============================================================================
def evaluate_imperceptibility():
    print("正在加载 LPIPS 模型...")
    loss_fn_lpips = lpips.LPIPS(net='alex').to(DEVICE)
    print("模型加载完成。\n")

    result_folders = sorted(glob.glob(os.path.join(RESULTS_BASE_DIR, "*/")))

    if not result_folders:
        print(f"错误: 在 '{RESULTS_BASE_DIR}' 中没有找到任何子文件夹。")
        return

    # 存储指标
    scores = {
        'psnr': [],
        'ssim': [],
        'lpips': []
    }

    print(f"开始评估，共找到 {len(result_folders)} 个样本文件夹...")

    # 使用 tqdm 显示进度条
    for folder in tqdm(result_folders, desc="评估进度"):
        folder_name = os.path.basename(os.path.normpath(folder))

        ori_path = os.path.join(folder, "ori.png")
        adv_path = os.path.join(folder, "adv.png")
        res_path = os.path.join(folder, "res.png")

        if not (os.path.exists(ori_path) and os.path.exists(adv_path)):
            continue

        # 读取图像为 RGB Numpy 数组
        ori_img = np.array(Image.open(ori_path).convert("RGB"))
        adv_img = np.array(Image.open(adv_path).convert("RGB"))

        # 1. 计算 PSNR
        psnr_val = compare_psnr(ori_img, adv_img, data_range=255)
        scores['psnr'].append(psnr_val)

        # 2. 计算 SSIM (指定 channel_axis=2 处理 RGB 图像)
        ssim_val = compare_ssim(ori_img, adv_img, multichannel=True, data_range=255, channel_axis=2)
        scores['ssim'].append(ssim_val)

        # 3. 计算 LPIPS
        ori_tensor = preprocess_image_for_lpips(ori_img, DEVICE)
        adv_tensor = preprocess_image_for_lpips(adv_img, DEVICE)
        with torch.no_grad():
            lpips_val = loss_fn_lpips(ori_tensor, adv_tensor).item()
        scores['lpips'].append(lpips_val)

        # 4. 生成并保存残差热力图
        generate_and_save_residual_map(ori_img, adv_img, res_path)

    # --- 打印最终汇总报告 ---
    if not scores['psnr']:
        print("未成功处理任何图像对，请检查路径和文件是否存在。")
        return

    print("\n" + "=" * 50)
    print(f" 扰动不可感知性 (Imperceptibility) 评估报告")
    print("=" * 50)
    print(f"评估样本总数 : {len(scores['psnr'])}")
    print(f"结果文件夹   : {RESULTS_BASE_DIR}")
    print("-" * 50)
    print(f"平均 PSNR    : {np.mean(scores['psnr']):.2f} dB  (越高越好，通常 >30dB 视觉无损)")
    print(f"平均 SSIM    : {np.mean(scores['ssim']):.4f}      (越高越好，接近 1.0 表示结构一致)")
    print(f"平均 LPIPS   : {np.mean(scores['lpips']):.4f}      (越低越好，表示感知差异越小)")
    print("-" * 50)
    print("提示: 残差热力图 (res.png) 已生成在各个样本的文件夹中。")
    print("      深蓝色表示未修改区域，红色表示噪声强度最大的区域。")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_imperceptibility()