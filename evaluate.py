import os
import glob
import random
import numpy as np
import torch
from PIL import Image
import cv2

from evaluations.FaceImageQuality.face_image_quality import SER_FIQ

import insightface
from insightface.app import FaceAnalysis
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from scipy.spatial.distance import cosine

import lpips
from torch_fidelity import calculate_metrics

from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from diffusers.utils import load_image
from ip_adapter import IPAdapter

# ==============================================================================
# 1. 配置区域
# ==============================================================================

# results
RESULTS_BASE_DIR = "/home/lab/workspace/works/cyx/faceshield-main/results/result_2025-12-04-17-09-02"

# 存放用于换脸的目标图像的文件夹
TARGET_IMAGES_DIR = "/home/lab/workspace/works/cyx/faceshield-main/results/result_2025-12-15-10-57-33"

# 存放评估过程中生成的换脸结果的临时文件夹
EVAL_OUTPUT_DIR = "evaluation_outputs/"
result_folder_name = os.path.basename(os.path.normpath(RESULTS_BASE_DIR))
EVAL_CLEAN_DIR = os.path.join(EVAL_OUTPUT_DIR, f"clean_swaps_{result_folder_name}_faceonly")
EVAL_PROTECTED_DIR = os.path.join(EVAL_OUTPUT_DIR, f"protected_swaps_{result_folder_name}_faceonly")

BASE_MODEL_PATH = "/home/lab/workspace/works/cyx/ZoDiac-master/ZoDiac-master/stable-diffusion-v1-5"
VAE_MODEL_PATH = "stabilityai/sd-vae-ft-mse"
IMAGE_ENCODER_PATH = "utils/image_encoder/"
IP_CKPT_PATH = "utils/unet/ip_adapter/ip-adapter_sd15.bin"
DEVICE = "cuda"

# ======================= 新增辅助函数：人脸裁剪与缩放 =======================
def crop_face_and_resize(image_np, face_app, target_size=(256, 256)):
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image_np

    faces = face_app.get(img_bgr)
    if not faces:
        return None

    face = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[0]
    box = face.bbox.astype(int)

    h, w, _ = image_np.shape
    x1 = max(0, box[0])
    y1 = max(0, box[1])
    x2 = min(w, box[2])
    y2 = min(h, box[3])

    if x1 >= x2 or y1 >= y2:
        return None

    face_crop = image_np[y1:y2, x1:x2]

    face_crop_resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
    return face_crop_resized
# ===========================================================================

# ==============================================================================
# 2. 度量指标计算函数
# ==============================================================================
def calculate_normalized_l2_mse(img1_np, img2_np):
    img1_norm = img1_np.astype(np.float32) / 255.0
    img2_norm = img2_np.astype(np.float32) / 255.0

    return np.mean((img1_norm - img2_norm) ** 2)


def get_face_embedding(image_np, face_analyzer):
    faces = face_analyzer.get(image_np)
    if not faces:
        return None
    return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[
        0].normed_embedding


def calculate_ism(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0
    return 1 - cosine(embedding1, embedding2)

def preprocess_image_for_lpips(image_np, device):
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0

def calculate_ser_fiq_for_folder(folder_path, ser_fiq_model):
    scores = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return 0

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # SER-FIQ 需要对齐后的人脸
        aligned_img = ser_fiq_model.apply_mtcnn(img)
        if aligned_img is not None:
            # T=100 是 SER-FIQ 论文中推荐的设置
            score = ser_fiq_model.get_score(aligned_img, T=100)
            scores.append(score)

    return np.mean(scores) if scores else 0
# ==============================================================================
# 3. 核心换脸与辅助函数
# ==============================================================================

def create_feathered_mask(image_size, face_bbox, blur_amount=30, expand_factor=1.3):
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    x1, y1, x2, y2 = [int(b) for b in face_bbox]
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    width, height = x2 - x1, y2 - y1
    new_width, new_height = int(width * expand_factor), int(height * expand_factor)

    cv2.ellipse(mask, (center_x, center_y), (new_width // 2, new_height // 2), 0, 0, 360, (255), -1)

    if blur_amount > 0:
        blur_ksize = int(blur_amount * 2) + 1
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    return Image.fromarray(mask)


def face_swap(source_face_path, target_image_path, output_path, ip_adapter_model, face_detector, **kwargs):
    source_face_image = load_image(source_face_path)
    target_image = load_image(target_image_path)

    target_cv_image = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
    faces = face_detector.get(target_cv_image)

    if not faces:
        print(f"警告: 在目标图像 {target_image_path} 中未检测到人脸，跳过此次换脸。")
        return False

    largest_face = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[0]
    face_bbox = largest_face.bbox
    mask_image = create_feathered_mask(target_image.size, face_bbox)

    images = ip_adapter_model.generate(
        pil_image=source_face_image, image=target_image, mask_image=mask_image, **kwargs
    )
    images[0].save(output_path)
    output_dir = os.path.dirname(output_path)
    target_save_dir = os.path.join(output_dir, "target")
    os.makedirs(target_save_dir, exist_ok=True)

    target_save_name = os.path.basename(output_path)
    target_image.save(os.path.join(target_save_dir, target_save_name))
    return True


# ==============================================================================
# 4. 主评估流程
# ==============================================================================

def evaluate_protection():
    # --- 模型加载 ---
    print("正在加载评估模型...")
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(512, 512))
    loss_fn_lpips = lpips.LPIPS(net='alex').to(DEVICE)
    # 加载 SER-FIQ 模型
    ser_fiq_model = SER_FIQ(gpu=None)

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                                    beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
                                    steps_offset=1)
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH).to(dtype=torch.float16)
    pipeline = StableDiffusionInpaintPipelineLegacy.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16,
                                                                    scheduler=noise_scheduler, vae=vae,
                                                                    feature_extractor=None, safety_checker=None)
    ip_model = IPAdapter(pipeline, IMAGE_ENCODER_PATH, IP_CKPT_PATH, DEVICE)
    print("所有模型加载完毕！\n")

    # --- 准备工作 ---
    os.makedirs(EVAL_CLEAN_DIR, exist_ok=True)
    os.makedirs(EVAL_PROTECTED_DIR, exist_ok=True)
    result_folders = sorted(glob.glob(os.path.join(RESULTS_BASE_DIR, "*/")))
    target_image_list = [os.path.join(TARGET_IMAGES_DIR, f) for f in os.listdir(TARGET_IMAGES_DIR)]

    if not result_folders:
        print(f"错误: 在 '{RESULTS_BASE_DIR}' 中没有找到任何结果子文件夹。")
        return
    if not target_image_list:
        print(f"错误: 在 '{TARGET_IMAGES_DIR}' 中没有找到任何目标图像。")
        return

    # --- 存储所有评估结果 ---
    all_scores = {
        'l2': [], 'psnr': [], 'ssim': [], 'lpips': [],
        'ism_insightface': [], 'ism_protected_insightface': [],
        'vis_psnr': [], 'vis_ssim': [],
    }

    # FDFR 需要单独计数
    clean_detection_failures = 0
    protected_detection_failures = 0

    # --- 遍历每个结果文件夹进行测试 ---
    for i, folder in enumerate(result_folders):
        folder_name = os.path.basename(os.path.normpath(folder))
        print(f"--- [{i + 1}/{len(result_folders)}] 正在测试: {folder_name} ---")

        source_path = os.path.join(folder, "ori.png")
        protected_path = os.path.join(folder, "adv.png")

        if not (os.path.exists(source_path) and os.path.exists(protected_path)):
            print(f"警告: 在 {folder_name} 中缺少 ori.png 或 adv.png，已跳过。")
            continue

        target_path = random.choice(target_image_list)
        clean_swapped_path = os.path.join(EVAL_CLEAN_DIR, f"{folder_name}_swap.png")
        protected_swapped_path = os.path.join(EVAL_PROTECTED_DIR, f"{folder_name}_swap.png")

        # --- 执行换脸 ---
        success_clean = face_swap(source_path, target_path, clean_swapped_path, ip_model, face_app, num_samples=1,
                                  num_inference_steps=30, seed=42, strength=0.7)
        success_protected = face_swap(protected_path, target_path, protected_swapped_path, ip_model, face_app,
                                      num_samples=1, num_inference_steps=30, seed=42, strength=0.7)

        if not (success_clean and success_protected):
            print("由于换脸失败，跳过此次评估。")
            continue

        # --- 计算指标 ---
        print("换脸完成，正在计算评估指标...")
        clean_swap_img = np.array(Image.open(clean_swapped_path).convert("RGB"))
        protected_swap_img = np.array(Image.open(protected_swapped_path).convert("RGB"))

        # 1. 先裁剪出人脸区域
        clean_face_crop = crop_face_and_resize(clean_swap_img, face_app, target_size=(256, 256))
        protected_face_crop = crop_face_and_resize(protected_swap_img, face_app, target_size=(256, 256))
        # 2. 仅当两张图都成功检测到人脸时，才计算失真指标
        if clean_face_crop is not None and protected_face_crop is not None:
            # L2, PSNR, SSIM, LPIPS (现在的输入是 crop 后的图)
            all_scores['l2'].append(calculate_normalized_l2_mse(clean_face_crop, protected_face_crop))
            all_scores['psnr'].append(compare_psnr(clean_face_crop, protected_face_crop, data_range=255))
            all_scores['ssim'].append(
                compare_ssim(clean_face_crop, protected_face_crop, multichannel=True, data_range=255, channel_axis=2))

            clean_tensor = preprocess_image_for_lpips(clean_face_crop, DEVICE)
            protected_tensor = preprocess_image_for_lpips(protected_face_crop, DEVICE)
            all_scores['lpips'].append(loss_fn_lpips(clean_tensor, protected_tensor).item())

            print(
                f"Face-Region -> L2: {all_scores['l2'][-1]:.2f} | PSNR: {all_scores['psnr'][-1]:.2f} | SSIM: {all_scores['ssim'][-1]:.4f} | LPIPS: {all_scores['lpips'][-1]:.4f}")
        else:
            print("警告: 无法在换脸结果中检测到人脸，跳过此次失真度指标计算。")

        source_embedding_insightface = get_face_embedding(cv2.imread(source_path), face_app)
        clean_swap_embedding_insightface = get_face_embedding(cv2.cvtColor(clean_swap_img, cv2.COLOR_RGB2BGR), face_app)
        protected_swap_embedding_insightface = get_face_embedding(cv2.cvtColor(protected_swap_img, cv2.COLOR_RGB2BGR),
                                                                  face_app)

        ism_clean_insightface = calculate_ism(source_embedding_insightface, clean_swap_embedding_insightface)
        ism_protected_insightface = calculate_ism(source_embedding_insightface, protected_swap_embedding_insightface)
        all_scores['ism_insightface'].append(ism_clean_insightface)
        all_scores['ism_protected_insightface'].append(ism_protected_insightface)

        print(
            f"ISM Clean: {all_scores['ism_insightface'][-1]:.4f} | ISM Protected: {all_scores['ism_protected_insightface'][-1]:.4f}↓\n")

        print(f"身份保护 (InsightFace) -> ISM Clean: {ism_clean_insightface:.4f} | ISM Protected: {ism_protected_insightface:.4f}↓")

        source_img = np.array(Image.open(source_path).convert("RGB"))
        protected_img = np.array(Image.open(protected_path).convert("RGB"))
        # 计算扰动可见性
        vis_psnr = compare_psnr(source_img, protected_img, data_range=255)
        vis_ssim = compare_ssim(source_img, protected_img, multichannel=True, data_range=255, channel_axis=2)
        all_scores['vis_psnr'].append(vis_psnr)
        all_scores['vis_ssim'].append(vis_ssim)
        print(f"可见性 -> PSNR: {vis_psnr:.2f}↑ | SSIM: {vis_ssim:.4f}↑ (越高代表扰动越不可见)")

    # --- 计算 FID ---
    print("\n--- 正在计算 FID 分数 (可能需要一些时间) ---")
    if not all_scores['l2']:
        print("未能成功完成任何评估，跳过 FID 计算。")
        fid_score = float('nan')
    else:
        metrics_dict = calculate_metrics(
            input1=EVAL_CLEAN_DIR,
            input2=EVAL_PROTECTED_DIR,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False
        )
        fid_score = metrics_dict['frechet_inception_distance']
        print(f"FID 计算完成: {fid_score:.2f}")

        print("正在计算干净输出的 SER-FIQ 分数...")
        clean_fiq_score = calculate_ser_fiq_for_folder(EVAL_CLEAN_DIR, ser_fiq_model)
        print(f"干净输出的 SER-FIQ 分数: {clean_fiq_score:.4f}")

        print("正在计算受保护输出的 SER-FIQ 分数...")
        protected_fiq_score = calculate_ser_fiq_for_folder(EVAL_PROTECTED_DIR, ser_fiq_model)
        print(f"受保护输出的 SER-FIQ 分数: {protected_fiq_score:.4f}")

    # --- 汇总报告 ---
    print("\n==================================== 最终评估总结报告 =====================================")
    if not all_scores['l2']:
        print("未能成功完成任何评估。")
        return

    num_samples = len(all_scores['l2'])
    clean_fdfr = clean_detection_failures / num_samples
    protected_fdfr = protected_detection_failures / num_samples

    print(f"总计成功测试样本数: {num_samples}")
    print("-" * 85)

    print(f"【扰动可见性评估】 (比较 保护图像 vs. 原始图像):")
    print(f"  - 平均 PSNR: {np.mean(all_scores['vis_psnr']):.2f} dB  (越高越好 ↑, 代表扰动在视觉上越隐蔽)")
    print(f"  - 平均 SSIM: {np.mean(all_scores['vis_ssim']):.4f}    (越高越好 ↑, 代表结构更接近原始图像)")
    print("-" * 85)
    print(f"【防御效果评估 - 图像质量与人脸可检测性】:")
    print(f"  - 平均 SER-FIQ (干净)   : {clean_fiq_score:.2f} (基准线, 代表无保护时的伪造人脸质量)")
    print(f"  - 平均 SER-FIQ (保护) : {protected_fiq_score:.2f} (越低越好 ↓, 代表人脸质量被破坏)")
    print(f"  - FDFR (干净)         : {clean_fdfr:.2%} (基准线, 代表检测器在无保护伪造图上的失败率)")
    print(f"  - FDFR (保护)         : {protected_fdfr:.2%} (越高越好 ↑, 代表保护成功地让检测器失效)")
    print("-" * 85)
    print(f"【防御效果评估 - 图像质量/失真度】 (比较 保护输出 vs. 干净输出):")
    print(f"  - 平均 MSE (L2) : {np.mean(all_scores['l2']):.4f}   (越高越好 ↑, 代表防御造成的失真越大)")
    print(f"  - 平均 PSNR     : {np.mean(all_scores['psnr']):.2f} dB (越低越好 ↓, 代表防御造成的失真越大)")
    print(f"  - 平均 SSIM     : {np.mean(all_scores['ssim']):.4f}   (越低越好 ↓, 代表结构差异越大)")
    print(f"  - 平均 LPIPS    : {np.mean(all_scores['lpips']):.4f}   (越高越好 ↑, 代表感知差异越大)")
    print("-" * 85)

    print(f"【防御效果评估 - 身份保护】 (比较 源脸 vs. 伪造输出):")
    print(f"  --- 基于 InsightFace ---")
    print(f"  - 平均 ISM (干净)   : {np.mean(all_scores['ism_insightface']):.4f} (基准线)")
    print(f"  - 平均 ISM (保护) : {np.mean(all_scores['ism_protected_insightface']):.4f} (越低越好 ↓)")
    print(f"  - FID 分数      : {fid_score:.2f}   (越高代表保护使输出分布偏移越大，效果越好 ↑)")
    print("===========================================================================================")


if __name__ == "__main__":
    evaluate_protection()