import os
import dlib
import cv2
import numpy as np
import pywt

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import datetime

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL

# IP-Adapter
from utils.unet.ip_adapter.ip_adapter import ImageProjModel
from utils.unet.ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from utils.unet.ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, \
        AttnProcessor2_0 as AttnProcessor
else:
    from utils.unet.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

# Utility functions
from utils.utils import (
    get_loss_function, compute_vae_encodings, face_detection_mask, get_filelist,
    merge_image, save_png, scale_tensor, create_line_mask, apply_gaussian, AttentionStore
)

# Attack modules
from utils.unet.unet_attack import AttackUnet_IP_all, AttackCLIP
from utils.landmark.mtcnn_attack import mtcnn_attack
from utils.landmark.arcface_attack import AttackArcFace

# DCT tools
from utils.dct import dct_pass_filter, make_dct_basis, blockfy, encode, decode, deblockfy


# 方差计算
def variance_attention_loss(attn_maps, semantic_masks, device):
    """
    计算基于区域方差的注意力损失 (L_attn)。
    """
    total_loss = torch.tensor(0.0, device=device)
    valid_layers = 0

    # 遍历 U-Net 返回的所有 Attention Maps
    for attn_output in attn_maps:
        # attn_output 可能是 (attn_map,) 元组
        A_map = attn_output[0] if isinstance(attn_output, tuple) else attn_output

        # A_map shape: (Batch*Heads, Res, Seq_len)
        if A_map.ndim != 3:
            continue

        b_h, res, seq = A_map.shape

        # === 1. 筛选 Cross-Attention 层 ===
        if seq == res:
            continue  # 跳过 Self-Attention

        # === 2. 计算方差 sigma_var ===
        sigma_var = torch.var(A_map, dim=-1, unbiased=False)

        # === 3. 计算最大方差 sigma_max ===
        seq_f = float(seq)
        sigma_max_val = (1.0 / seq_f) * ((1.0 - 1.0 / seq_f) ** 2 + (seq_f - 1.0) / (seq_f ** 2))
        sigma_max = torch.tensor(sigma_max_val, device=device)

        # === 4. 处理语义掩码 M ===
        side = int(np.sqrt(res))
        if side * side != res:
            continue

        combined_mask = torch.zeros((side, side), device=device)
        if semantic_masks is not None:
            for _, mask in semantic_masks.items():
                resized_mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(side, side),
                    mode='nearest'
                ).squeeze()
                combined_mask = torch.max(combined_mask, resized_mask)

        M = combined_mask.view(-1)
        M = M.expand_as(sigma_var)

        # === 5. 计算损失 ===
        diff = sigma_var * M
        layer_loss = torch.sum(diff ** 2) / (torch.sum(M) + 1e-6)

        total_loss += layer_loss
        valid_layers += 1

    return total_loss / (valid_layers + 1e-6)

def get_facial_semantic_masks(face_tensor, predictor, device):

    image = (face_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # dlib需要RGB
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)

    if len(rects) == 0:
        print("Warning: Dlib did not detect a face. Semantic attention loss will be disabled.")
        return None

    shape = predictor(image_rgb, rects[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    eye_left_indices = list(range(36, 42))
    eye_right_indices = list(range(42, 48))
    nose_indices = list(range(27, 36))
    mouth_indices = list(range(48, 68))

    masks = {}
    for name, indices in [("eyes", eye_left_indices + eye_right_indices),
                          ("nose", nose_indices),
                          ("mouth", mouth_indices)]:
        points = landmarks[indices]
        hull = cv2.convexHull(points)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        cv2.fillConvexPoly(mask, hull, 1)
        masks[name] = torch.from_numpy(mask).to(device)

    return masks

class FeatureExtractor:
    def __init__(self):
        self.features = {}
        self.hooks = []

    def register_hooks(self, model, layer_names):
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)

    def _get_hook(self, name):
        def hook(model, input, output):
            self.features[name] = output[0] if isinstance(output, tuple) else output

        return hook

    def clear(self):
        self.features = {}
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class SpatialFeatureLoss(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.criterion = torch.nn.MSELoss()

    def normalize(self, x):

        if x.ndim == 4:
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True) + 1e-6

        elif x.ndim == 2:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-6

        elif x.ndim == 3:
            mean = x.mean(dim=1, keepdim=True)  # 通常在 Tokens 维度归一化，或者 dim=(1,2) 全局归一化
            std = x.std(dim=1, keepdim=True) + 1e-6

        else:
            return x

        return (x - mean) / std

    def forward(self, adv_feat, gt_feat):

        adv_feat = adv_feat.to(torch.float32)
        gt_feat = gt_feat.to(torch.float32)

        if torch.isnan(adv_feat).any() or torch.isinf(adv_feat).any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        adv_feat_norm = self.normalize(adv_feat)
        gt_feat_norm = self.normalize(gt_feat)


        distance = self.criterion(adv_feat_norm, gt_feat_norm)
        return -distance


def get_parser():
    parser = argparse.ArgumentParser()

    # Seed & model paths
    parser.add_argument("--seed", type=int, default=1, help="Random seed (used with seed_everything)")
    parser.add_argument("--model_path", type=str, default="/home/lab/workspace/works/cyx/ZoDiac-master/ZoDiac-master/stable-diffusion-v1-5",
                        help="Path or name of pretrained Stable Diffusion model")
    parser.add_argument("--vae_model_path", type=str, default="stabilityai/sd-vae-ft-mse", help="Path or name of pretrained VAE model")
    parser.add_argument("--unet_config", type=str, default="utils/unet/unet_config15.json", help="YAML config for attack UNet")
    parser.add_argument("--pretrained_ip_adapter_path", type=str, default="utils/unet/ip_adapter/ip-adapter_sd15.bin",
                        help="Path to pretrained IP-Adapter weights")
    parser.add_argument("--image_encoder_path", type=str, default="h94/IP-Adapter",
                        help="Path to pretrained image encoder (e.g., CLIP)")
    parser.add_argument("--pretrained_arcface50_path", type=str, default="models/arcface50_checkpoint.tar",
                        help="Path to pretrained ArcFace-50 model")
    parser.add_argument("--pretrained_arcface100_path", type=str, default="models/arcface100_checkpoint.tar",
                        help="Path to pretrained ArcFace-100 model")

    # FaceShield settings
    parser.add_argument("--save_path", type=str, default="results", help="Directory to save results")
    parser.add_argument("--resize_shape", type=int, default=512, help="Resize image to this square resolution")
    parser.add_argument("--proj_func", type=str, default="l1", help="Loss function for projection loss (CLIP loss)")
    parser.add_argument("--mtcnn_func", type=str, default=False, help="Loss function for MTCNN feature loss")
    parser.add_argument("--arc_func", type=str, default="cosine", help="Loss function for ArcFace feature loss")
    parser.add_argument("--attn_threshold", type=float, default=0.2, help="Threshold for masking low-attention regions")
    parser.add_argument("--total_iter", type=int, default=30, help="Number of PGD iterations")
    parser.add_argument("--noise_clamp", type=int, default=12, help="Clamp value for adversarial noise (L∞ norm)")
    parser.add_argument("--step_size", type=float, default=1., help="Step size per PGD iteration")
    parser.add_argument("--image_path", type=str, default="data/src1", help="Input image or directory path")

    parser.add_argument("--dlib_predictor_path", type=str, default="utils/shape_predictor_68_face_landmarks.dat",
                        help="Path to dlib's facial landmark predictor model.")
    parser.add_argument("--num_attack_layers", type=int, default=5, help="Number of random U-Net layers to attack.")
    parser.add_argument("--lambda_mtcnn", type=float, default=9.0, help="Weight for MTCNN loss.")
    # default 5.0
    parser.add_argument("--lambda_arc", type=float, default=5.0, help="Weight for ArcFace identity loss.")
    parser.add_argument("--lambda_clip", type=float, default=1.0, help="Weight for CLIP image encoder loss.")
    # default 0.5
    parser.add_argument("--lambda_semantic", type=float, default=0.5,
                        help="Weight for the new semantic attention loss.")
    # default 0.2
    parser.add_argument("--lambda_layer", type=float, default=0.2,
                        help="Weight for the new randomized U-Net layer loss.")
    # default 0.1
    parser.add_argument("--lambda_wavelet", type=float, default=0.01, help="Weight for the new wavelet domain loss.")

    return parser


def attack(args, gpu_num, gpu_no, **kwargs):
    config = OmegaConf.load(args.unet_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
    unet = AttackUnet_IP_all.from_pretrained(args.model_path, subfolder="unet", config_file=config, strict=False)
    image_preprocess = AttackCLIP()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path,
                                                                  subfolder="models/image_encoder")
    face_embedder50 = torch.load(args.pretrained_arcface50_path, weights_only=False)
    face_embedder100 = torch.load(args.pretrained_arcface100_path, weights_only=False)
    id_preprocess = AttackArcFace()

    # Freeze parameters of models to save more memory
    vae.requires_grad_(False).to(device)
    text_encoder.requires_grad_(False).to(device)
    image_encoder.requires_grad_(False).to(device)
    face_embedder50.requires_grad_(False).to(device)
    face_embedder100.requires_grad_(False).to(device)

    # ================================================== IP_Adapter =============================================== #
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    ).to(device)

    # Init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        elif name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    # Calculate original checksums
    orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in image_proj_model.parameters()]))
    orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in adapter_modules.parameters()]))

    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location=device, weights_only=True)

    # Load state dict for image_proj_model and adapter_modules
    image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
    adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

    # Calculate new checksums
    new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in image_proj_model.parameters()]))
    new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in adapter_modules.parameters()]))

    # Verify if the weights have changed
    assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
    assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

    unet.requires_grad_(False).to(device)
    # ============================================================================================================== #
    # Text token
    inputs = tokenizer([""], max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                       return_tensors="pt")
    encoder_hidden_states = text_encoder(inputs.input_ids.to(device))[0]

    # Source Image and Mask load
    dataset_name = args.image_path.split('/')[-2]
    src_list = get_filelist(args.image_path, ['png', 'jpg'])
    if len(src_list) == 0:
        src_list = [args.image_path]
    num_samples = len(src_list)

    # Multi-gpu setting
    samples_split = num_samples // gpu_num
    remainder = num_samples % gpu_num
    if gpu_no < remainder:
        start_idx = gpu_no * (samples_split + 1)
        end_idx = start_idx + samples_split + 1
    else:
        start_idx = gpu_no * samples_split + remainder
        end_idx = start_idx + samples_split

    indices = list(range(start_idx, end_idx))
    gpu_samples = len(indices)
    src_list_rank = [src_list[i] for i in indices]
    filename_list = [f"{os.path.split(src_list_rank[id])[-1][:-4]}" for id in range(gpu_samples)]

    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_output_dir = os.path.join(args.save_path, f"result_{run_timestamp}")
    print(f"Output directory: {base_output_dir}")

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for indice in range(gpu_samples):
            batchT = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            current_img_name = filename_list[indice]
            save_dir = os.path.join(base_output_dir, current_img_name)
            os.makedirs(save_dir, exist_ok=True)

            face, _, _, back, coor = face_detection_mask(src_list_rank[indice], save_dir, args.resize_shape, device)
            if face is None:
                print(f"\n[警告] 在图像 '{os.path.basename(src_list_rank[indice])}' 中未能检测到人脸。")
                print("       --> 已跳过此图像，继续处理下一个。\n")
                continue
            gt_face = face.clone().detach()
            save_png(os.path.join(save_dir, "ori"), gt_face.squeeze(0))

            # Loss Functions
            proj_func = get_loss_function(args.proj_func)
            mtcnn_func = get_loss_function(args.mtcnn_func)
            arc_func = get_loss_function(args.arc_func)

            # wavelet_loss_fn = WaveletLoss(device)  # 创新点3
            spatial_loss_fn = SpatialFeatureLoss(device)

            latents_query = compute_vae_encodings(face, vae, device, gt=True)

            N = 8
            DCT_basis = make_dct_basis(N, device)
            low_pass_filter, high_pass_filter = dct_pass_filter(device)
            timestep = torch.tensor([0], device=device)

            # ArcFace GT
            gt_50, gt_100 = id_preprocess.preprocess(gt_face)
            gt_id_50 = face_embedder50(gt_50.to(device))
            gt_id_100 = face_embedder50(gt_100.to(device))


            print("Generating semantic facial masks...")
            dlib_predictor = dlib.shape_predictor(args.dlib_predictor_path)
            semantic_masks = get_facial_semantic_masks(gt_face, dlib_predictor, device)

            print(f"Selecting {args.num_attack_layers} random U-Net layers to attack...")
            possible_layers = [name for name, module in unet.named_modules() if
                               'down_blocks' in name and 'resnets' in name]
            attack_layer_names = np.random.choice(possible_layers, args.num_attack_layers, replace=False).tolist()
            print(f"Attacking layers: {attack_layer_names}")

            # UNet GT (CLIP encoding)
            gt_preprocessed = image_preprocess(gt_face)
            gt_encoded = image_encoder(gt_preprocessed).image_embeds

            var_controller = AttentionStore()
            clean_feature_extractor = FeatureExtractor()
            clean_feature_extractor.register_hooks(unet, attack_layer_names)
            gt_proj = image_proj_model(gt_encoded)
            stacked_gt_hidden_states = torch.cat([encoder_hidden_states, gt_proj], dim=1)
            unet(latents_query, timestep, stacked_gt_hidden_states, store_controller=var_controller, unet_threshold=args.attn_threshold)

            clean_layer_outputs = {name: feat.clone().detach() for name, feat in
                                   clean_feature_extractor.features.items()}
            clean_feature_extractor.clear()

            with torch.enable_grad():
                delta = torch.zeros_like(gt_face, requires_grad=True).to(device)
                epochs = tqdm(range(args.total_iter), position=(indice * gpu_num + gpu_no),
                              desc=f"[rank:{gpu_no}] batch {batchT}: {indice + 1}/{gpu_samples}",
                              total=len(range(args.total_iter)))
                for i, _ in enumerate(epochs):
                    adv_face = (255 * gt_face) + delta
                    adv_face = torch.clamp(adv_face, min=0, max=255)

                    # 1. MTCNN attack
                    mtcnn_loss = torch.tensor(0.0, device=device)
                    mtcnn_loss = mtcnn_attack(2 * (adv_face / 255) - 1, loss_fn=mtcnn_func, loss=mtcnn_loss,
                                              device=device)

                    # 2. ArcFace Identity Attack
                    adv_50, adv_100 = id_preprocess.preprocess(adv_face / 255)
                    adv_id_50 = face_embedder50(adv_50)
                    adv_id_100 = face_embedder100(adv_100)
                    id_loss_50 = arc_func(adv_id_50, gt_id_50)
                    id_loss_100 = arc_func(adv_id_100, gt_id_100)
                    id_loss = (-1) * id_loss_50 + (-1) * id_loss_100

                    # --- Diff-Conditioned Attacks ---
                    adv_preprocessed = image_preprocess(adv_face / 255)
                    adv_encoded = image_encoder(adv_preprocessed).image_embeds
                    adv_proj = image_proj_model(adv_encoded)
                    stacked_adv_hidden_states = torch.cat([encoder_hidden_states, adv_proj], dim=1)

                    clip_loss = proj_func(adv_encoded, gt_encoded)

                    # --- U-Net Forward Pass for New Losses ---
                    adv_attn_controller = AttentionStore()
                    adv_feature_extractor = FeatureExtractor()
                    adv_feature_extractor.register_hooks(unet, attack_layer_names)

                    unet(latents_query, timestep, stacked_adv_hidden_states, store_controller=adv_attn_controller, unet_threshold=args.attn_threshold)

                    semantic_attn_loss = torch.tensor(0.0, device=device)
                    if semantic_masks is not None:
                        semantic_attn_loss = variance_attention_loss(
                            adv_attn_controller.attn_map,
                            semantic_masks,
                            device
                        )
                    else:
                        semantic_attn_loss = torch.tensor(0.0, device=device)

                    layer_loss = torch.tensor(0.0, device=device)
                    layer_loss_fn = torch.nn.MSELoss()
                    for name, adv_feat in adv_feature_extractor.features.items():
                        clean_feat = clean_layer_outputs[name]
                        layer_loss -= layer_loss_fn(adv_feat, clean_feat)

                    spatial_loss = torch.tensor(0.0, device=device)
                    valid_layers = 0

                    for name, adv_feat in adv_feature_extractor.features.items():
                        clean_feat = clean_layer_outputs[name]

                        layer_s_loss = spatial_loss_fn(adv_feat, clean_feat)

                        if not torch.isnan(layer_s_loss):
                            spatial_loss += layer_s_loss
                            valid_layers += 1

                    if valid_layers > 0:
                        spatial_loss = spatial_loss / valid_layers

                    adv_feature_extractor.clear()

                    # ========== PGD Update with Weighted Losses ========== #
                    total_loss = (args.lambda_mtcnn * mtcnn_loss +
                                  args.lambda_arc * id_loss +
                                  (-1) * args.lambda_clip * clip_loss +
                                  args.lambda_semantic * semantic_attn_loss +
                                  args.lambda_layer * layer_loss +
                                  args.lambda_wavelet * spatial_loss)

                    print(f"total: {total_loss.item():.4f}, mtcnn: {mtcnn_loss.item():.4f}, "
                          f"id: {id_loss.item():.4f}, clip: {clip_loss.item():.4f}, "
                          f"sem: {semantic_attn_loss.item():.4f}, layer: {layer_loss.item():.4f}, "
                          f"wav: {spatial_loss.item():.4f}")
                    epochs.set_postfix({
                        "total": f"{total_loss.item():.4f}",
                        "mtcnn": f"{mtcnn_loss.item():.4f}",
                        "id": f"{id_loss.item():.4f}",
                        "clip": f"{clip_loss.item():.4f}",
                        "sem": f"{semantic_attn_loss.item():.4f}",
                        "layer": f"{layer_loss.item():.4f}",
                        "wav": f"{spatial_loss.item():.4f}"
                    })

                    total_loss.backward(retain_graph=True)
                    new_delta = args.step_size * torch.sign(delta.grad)

                    d_rgb = scale_tensor(new_delta)
                    mask = create_line_mask(save_dir, d_rgb)
                    new_delta = apply_gaussian(save_dir, new_delta, mask, 9, 5)
                    delta.data -= new_delta
                    grad_block, pad_size = blockfy(delta.data, N)
                    grad_dct = encode(grad_block, DCT_basis)
                    grad_dct_passed = grad_dct * low_pass_filter.expand(grad_dct.shape)
                    grad_block_passed = decode(grad_dct_passed, DCT_basis)
                    delta.data = deblockfy(grad_block_passed, pad_size)
                    delta.data = torch.clamp(delta.data, min=-args.noise_clamp, max=args.noise_clamp)
                    delta.grad = None

                    del mtcnn_loss, clip_loss, id_loss, id_loss_50, id_loss_100, total_loss
                    del semantic_attn_loss, layer_loss, spatial_loss
                    # del semantic_attn_loss, layer_loss, wavelet_loss
                    torch.cuda.empty_cache()

            face = torch.clamp((gt_face * 255) + delta, 0, 255) / 255
            save_png(os.path.join(save_dir, "adv"), face.squeeze(0))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    attack(args, gpu_num, rank)