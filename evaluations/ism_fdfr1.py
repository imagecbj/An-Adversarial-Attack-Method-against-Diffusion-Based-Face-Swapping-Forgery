import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from deepface import DeepFace
from tqdm import tqdm


# =========================== 辅助函数 ===========================

def compute_face_embedding(img_path):
    """
    提取给定图像的人脸嵌入向量。
    使用 ArcFace 模型和 RetinaFace 检测器。
    """
    if not os.path.exists(img_path):
        return None

    try:
        # DeepFace.represent 返回一个列表
        resps = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            enforce_detection=True,  # 强制检测，如果没检测到会抛出异常
            detector_backend="retinaface",
            align=True
        )

        if len(resps) == 0:
            return None

        # 如果检测到多张脸，选择面积最大的一张
        if len(resps) > 1:
            resps.sort(key=lambda resp: resp["facial_area"]["h"] * resp["facial_area"]["w"], reverse=True)

        return np.array(resps[0]["embedding"])

    except ValueError:
        # ValueError 通常表示 "Face could not be detected"
        return None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def calculate_ism_fdfr(source_root_dir, swap_dir):
    """
    计算 ISM 和 FDFR。

    Args:
        source_root_dir (str): 存放原图结果的根目录 (例如: .../results/result_2025-...)
                               结构应该是: source_root_dir/图片名文件夹/ori.png
        swap_dir (str): 存放换脸结果的文件夹 (例如: .../evaluation_outputs/protected_swaps)
                        结构应该是: 图片名_swap.png
    """

    swap_images = [f for f in os.listdir(swap_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(swap_images)

    if total_images == 0:
        print("未在换脸文件夹中找到图片。")
        return 0, 0

    fail_detection_count = 0
    total_ism = 0
    valid_pair_count = 0

    print(f"开始评估: 共找到 {total_images} 张换脸图。")

    for swap_name in tqdm(swap_images):
        swap_path = os.path.join(swap_dir, swap_name)

        # 1. 解析对应的原图路径
        # 假设换脸图命名规则是: "{folder_name}_swap.png"
        # 而原图位于: source_root_dir/{folder_name}/ori.png
        folder_name = swap_name.replace("_swap.png", "").replace("_swap.jpg", "")

        ori_path = os.path.join(source_root_dir, folder_name, "ori.png")

        # 兼容性检查：如果找不到原图，尝试直接在根目录找（取决于你具体的保存结构）
        if not os.path.exists(ori_path):
            # 备用方案：也许原图直接叫 folder_name.png ? 这里根据你的实际情况调整
            # print(f"警告: 找不到对应的原图 {ori_path}，跳过。")
            continue

        # 2. 提取特征
        # 提取换脸图特征 (如果提取失败，FDFR + 1)
        swap_emb = compute_face_embedding(swap_path)
        if swap_emb is None:
            fail_detection_count += 1
            continue  # 检测不到人脸，无法计算相似度，跳过

        # 提取原图特征 (如果原图都提取不出特征，这组数据作废，不计入 FDFR)
        ori_emb = compute_face_embedding(ori_path)
        if ori_emb is None:
            print(f"警告: 原图 {ori_path} 未检测到人脸，跳过此对比较。")
            continue

        # 3. 计算余弦相似度 (ISM)
        # DeepFace 返回的是 numpy array，转为 Tensor 计算
        emb1 = torch.tensor(ori_emb)
        emb2 = torch.tensor(swap_emb)

        # Cosine Similarity
        similarity = F.cosine_similarity(emb1, emb2, dim=0).item()

        total_ism += similarity
        valid_pair_count += 1

    # 4. 汇总结果
    # ISM = 总相似度 / 成功检测且成功匹配的对数
    avg_ism = total_ism / valid_pair_count if valid_pair_count > 0 else 0

    # FDFR = 检测失败数 / 总图片数
    fdfr = fail_detection_count / total_images

    return avg_ism, fdfr


def parse_args():
    parser = argparse.ArgumentParser(description='Independent FDFR and ISM evaluation')

    parser.add_argument('--source_root', type=str,
                        default="/home/lab/workspace/works/cyx/faceshield-main/results/result_2025-11-26-17-44-32",
                        help='包含 ori.png 的结果根目录')

    parser.add_argument('--swap_dir', type=str,
                        default="/home/lab/workspace/works/cyx/faceshield-main/evaluation_outputs/protected_swaps",
                        help='包含换脸结果的目录 ')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f"原图目录 (Source): {args.source_root}")
    print(f"换脸目录 (Swap):   {args.swap_dir}")

    ism, fdfr = calculate_ism_fdfr(args.source_root, args.swap_dir)

    print("\n" + "=" * 40)
    print(f"评估结果:")
    print(f"FDFR (人脸检测失败率): {fdfr:.4%} (越高代表防御使得人脸无法被识别)")
    print(f"ISM  (身份相似度):     {ism:.4f}   (越低代表换脸后的身份越不像原主)")
    print("=" * 40)


if __name__ == '__main__':
    main()