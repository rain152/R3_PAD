import os
import torch
import yaml
import json
import pycolmap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from glob import glob
from shutil import rmtree
from argparse import ArgumentParser
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from scipy.spatial.transform import Rotation
from PIL import Image
from gaussian_splatting.train import *
from gaussian_splatting.scene.colmap_loader import read_points3D_binary
from gaussian_splatting.scene.dataset_readers import storePly
from gaussian_splatting.scene.cameras import Camera
from easydict import EasyDict
from utils_pose_est import ModelHelper, update_config, DefectDataset
from aupro import calculate_au_pro_au_roc
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from hloc import extract_features, match_features, pairs_from_retrieval, triangulation, localize_sfm
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from Retinex_UNet import RetinexUNet

classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

def parse_arguments():
    """解析命令行参数"""
    parser = ArgumentParser(description="Parameters of the LEGO training run")
    parser.add_argument("-c", "--classname", type=str, help="current class to run experiments on", default="01Gorilla")
    parser.add_argument("-seed", type=int, help="seed for random behavior", default=0)
    parser.add_argument("-iters", type=int, help="number of training iterations for 3DGS", default=15000)
    parser.add_argument("-skip_loc", action="store_true", help='skip localization')                 
    parser.add_argument("-skip_train", action="store_true", help='skip training 3dgs')                    
    parser.add_argument("-data_path", type=str, help="preprocessed dataset path", default="MAD-Sim/")         
    parser.add_argument("-n_match", type=int, default=5, help="num of matches for netvlad image retrieval")    
    parser.add_argument("-trainset_ratio", type=float, default=1.0, help="percentage of training samples")     
    parser.add_argument("-feature_ext", type=str, default='superpoint', help="feature extractor", choices=["superpoint", "aliked"])
    return parser.parse_args()

def setup_environment(args):
    """初始化环境和目录结构"""
    data_path = os.path.join(args.data_path, args.classname)
    result_dir = os.path.join("results", args.classname)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Saving results to: {result_dir}")
    return data_path, result_dir

def run_colmap_reconstruction(data_path):
    """执行COLMAP重建和定位"""
    scene_path = Path(data_path)
    image_path = scene_path/'images'
    outputs = scene_path/'outputs'
    
    # 清理并重建目录结构
    if os.path.exists(image_path):
        rmtree(image_path)
    os.makedirs(image_path)
    os.makedirs(outputs, exist_ok=True)  
    os.symlink('../train/', image_path/'train/')
    os.symlink('../test/', image_path/'test/')

    # 处理相机参数
    with open(os.path.join(scene_path, 'transforms.json')) as f: flist = json.load(f)
    first_img = Image.open(scene_path/'train/good/0.png')
    (w, h), cam_angle_x = first_img.size, flist['camera_angle_x']
    focal = 0.5*w/np.tan(0.5*cam_angle_x)
    camera_id = 1  # increment for each new camera
    rec = pycolmap.Reconstruction()
    rec.add_camera(pycolmap.Camera(model='SIMPLE_PINHOLE', width=w, height=h, params=[focal, w/2, h/2], camera_id=camera_id))
    for i, frame in enumerate(tqdm(flist['frames'])):
        c2w = np.array(frame["transform_matrix"])   # NeRF 'transform_matrix' is a camera-to-world transform
        c2w[:3, 1:3] *= -1  # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        w2c = np.linalg.inv(c2w)
        pose = pycolmap.Rigid3d(w2c[:3])
        im = pycolmap.Image(f'train/good/{i}.png', [], pose, camera_id, i+1)
        im.registered = True
        rec.add_image(im)
        frame["file_path"] = frame["file_path"][:-4]
    rec.write(outputs)
    with open(os.path.join(scene_path, 'transforms_train.json'), 'w') as f: f.write(json.dumps(flist))
    flist["frames"] = flist["frames"][::20]
    with open(os.path.join(scene_path, 'transforms_test.json'), 'w') as f: f.write(json.dumps(flist))

    # 准备查询列表
    query_list_with_intrinsics_path = scene_path / 'query_list_with_intrinsics.txt'
    intrinsics_str = f' SIMPLE_PINHOLE {w} {h} {focal} {w/2} {h/2}\n'
    with open(query_list_with_intrinsics_path, 'w') as fp:
        query_list_with_intrinsics = [x + intrinsics_str for x in glob('test/*/*.png', root_dir=image_path)]
        fp.writelines(query_list_with_intrinsics)

    return outputs, image_path, w, h, focal

def run_feature_matching(args, outputs, image_path, data_path):
    """执行特征提取和匹配"""
    # 配置特征提取器
    retrieval_conf = extract_features.confs["netvlad"]
    if args.feature_ext == 'superpoint':
        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["superpoint+lightglue"]
    else:
        feature_conf = extract_features.confs["aliked-n16"]
        matcher_conf = match_features.confs["aliked+lightglue"]

    # 计时开始
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # 生成匹配对
    ref_pairs = outputs/'ref_pairs.txt'
    global_descriptors = extract_features.main(retrieval_conf, image_path, outputs)
    features = extract_features.main(feature_conf, image_path, outputs)
    pairs_from_retrieval.main(global_descriptors, ref_pairs, num_matched=args.n_match, query_prefix='train', db_prefix='train')
    sfm_matches = match_features.main(matcher_conf, ref_pairs, feature_conf["output"], outputs)
    rec = triangulation.main(outputs, outputs, image_path, ref_pairs, features, sfm_matches)
    
    # 计时结束
    end.record()
    torch.cuda.synchronize()
    sfm_time = start.elapsed_time(end)
    
    # 保存点云
    rec.write(outputs)
    xyz, rgb, _ = read_points3D_binary(outputs/'points3D.bin')
    storePly(Path(data_path)/'points3d.ply', xyz, rgb)
    
    return rec, sfm_time

def localize_test_images(args, data_path, rec):
    """定位测试图像"""
    scene_path = Path(data_path)
    outputs = scene_path/'outputs'
    image_path = scene_path/'images'
    first_img = Image.open(scene_path/'train/good/1.png')
    with open(os.path.join(scene_path, 'transforms.json')) as f: flist = json.load(f)
    (w, h), cam_angle_x = first_img.size, flist['camera_angle_x']
    focal = 0.5*w/np.tan(0.5*cam_angle_x)
    query_list_with_intrinsics_path = scene_path / 'query_list_with_intrinsics.txt'
    intrinsics_str = f' SIMPLE_PINHOLE {w} {h} {focal} {w/2} {h/2}\n'
    with open(query_list_with_intrinsics_path, 'w') as fp:
        query_list_with_intrinsics = [x + intrinsics_str for x in glob('test/*/*.png', root_dir=image_path)]
        fp.writelines(query_list_with_intrinsics)

    if args.feature_ext == 'superpoint':
        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["superpoint+lightglue"]
    else:
        feature_conf = extract_features.confs["aliked-n16"]
        matcher_conf = match_features.confs["aliked+lightglue"]

    retrieval_conf = extract_features.confs["netvlad"]
    features = extract_features.main(feature_conf, image_path, outputs)
    global_descriptors = extract_features.main(retrieval_conf, image_path, outputs)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # 执行定位
    loc_pairs = outputs/'loc_pairs.txt'
    pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=args.n_match, query_prefix='test', db_prefix='train')
    loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf["output"], outputs)
    results = scene_path/'query_poses.txt'
    localize_sfm.main(
        rec,
        query_list_with_intrinsics_path,
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
    )  
    
    end.record()
    torch.cuda.synchronize()
    loc_time = start.elapsed_time(end)
    
    # 计算平均定位时间
    with open(results) as f:
        num_queries = sum(1 for _ in f)
    return loc_time / num_queries if num_queries > 0 else 0

def train_gaussian_model(args, data_path, result_dir):
    """训练3D高斯模型"""
    # 准备训练参数
    training_args = [
        "-w", "-s", data_path, "-m", result_dir,
        "--iterations", str(args.iters), 
        "--densification_interval", "1000"
    ]
    
    # 解析参数
    parser = ArgumentParser(description="3DGS Training Parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[args.iters])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, args.iters])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args_parsed = parser.parse_args(training_args)
    args_parsed.save_iterations.append(args_parsed.iterations)

    # 执行训练
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    safe_state(args_parsed.quiet, args.seed)
    torch.autograd.set_detect_anomaly(args_parsed.detect_anomaly)
    training(
        lp.extract(args_parsed), 
        op.extract(args_parsed), 
        pp.extract(args_parsed), 
        args_parsed.test_iterations, 
        args_parsed.save_iterations,
        args_parsed.checkpoint_iterations, 
        args_parsed.start_checkpoint, 
        args_parsed.debug_from,
        args.trainset_ratio
    )
    
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

import cv2
def white_bg(image):
    # 检查输入是否为 PyTorch 张量
    if isinstance(image, torch.Tensor):
        # 将 PyTorch 张量转换为 NumPy 数组
        image_np = image.cpu().numpy()
    else:
        raise TypeError("Input image must be a PyTorch tensor")

    # 确保图像是二维（灰度图）或三维（彩色图）
    if len(image_np.shape) == 3:  # 如果是彩色图
        if image_np.shape[0] == 3 or image_np.shape[0] == 1:  # CxHxW 格式
            image_np = image_np.transpose(1, 2, 0)  # 转换为 HxWxC 格式
        if image_np.shape[2] == 3:  # 如果是 RGB 图像
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
        elif image_np.shape[2] == 1:  # 如果是单通道图像
            gray = image_np.squeeze()  # 去除通道维度
    elif len(image_np.shape) == 2:  # 如果是灰度图
        gray = image_np
    else:
        raise ValueError("Input image must be 2D (grayscale) or 3D (color)")

    # Sobel 边缘检测
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)  # 合并水平和垂直方向的梯度
    sobel = np.uint8(sobel / sobel.max() * 255)  # 归一化并转换为 8 位图像

    # 将边缘部分提取为掩码
    edges = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)[1]  # 二值化处理

    # 使用填充算法填充背景区域
    h, w = edges.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled_edges = edges.copy()
    cv2.floodFill(filled_edges, mask, (0, 0), 255)

    # 反向填充以获得前景掩码
    foreground_mask = cv2.bitwise_not(filled_edges)

    # 将边缘添加到前景掩码中
    foreground_mask = cv2.bitwise_or(foreground_mask, edges)
    white_background = np.full_like(image_np, 1, dtype=np.float32)  # 白色背景
    foreground = cv2.bitwise_and(image_np, image_np, mask=foreground_mask)
    background = cv2.bitwise_and(white_background, white_background, mask=cv2.bitwise_not(foreground_mask))

    # 合并前景和背景
    result_np = cv2.add(foreground, background)

    # 将结果转换回 PyTorch 张量
    result_torch = torch.from_numpy(result_np.transpose(2, 0, 1)).float()  # 转换为 CxHxW 格式
    if image.is_cuda:
        result_torch = result_torch.cuda()  # 如果输入在 GPU 上，将结果也放到 GPU
    return result_torch

def synthesize_reference_images(args, data_path, result_dir):
    """合成伪参考图像"""
    # 初始化数据集
    trainset = DefectDataset(data_path, args.classname, "train", True, True)
    testset = DefectDataset(data_path, args.classname, "test", True, True)
    fov = trainset.camera_angle
    tanfov = np.tan(fov/2)
    bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")

    # 加载位姿数据
    test_RT = {}
    with open(Path(data_path)/args.classname/'query_poses.txt') as fp:
        for frame in fp.readlines():
            name, qw, qx, qy, qz, x, y, z = frame.split()
            qw, qx, qy, qz, x, y, z = map(float, [qw, qx, qy, qz, x, y, z])
            test_RT[name] = (Rotation.from_quat([qx, qy, qz, qw]).as_matrix().T, [x,y,z]) # R is stored transposed due to 'glm' in CUDA code

    # 加载高斯模型
    gs = GaussianModel(3)
    model_path = os.path.join(result_dir, f"point_cloud/iteration_{args.iters}/point_cloud.ply")
    # model_path = f"/home/rain152/code/PAD/ckpt/{args.classname}.ply"
    gs.load_ply(model_path)

    # 渲染循环
    syn_imgs, ref_imgs, all_labels, gt_masks = [], [], [], []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0

    with torch.no_grad():
        for idx in tqdm(range(len(testset)), desc="Rendering test views"):
            # 准备数据
            img, label, mask = testset[idx]
            im_key = '/'.join(testset.images[idx].split('/')[2:])
            
            # 记录时间
            start_event.record()
            
            # 准备相机参数
            R, t = test_RT[im_key]
            cur_view = Camera(
                colmap_id=1,
                R=R, 
                T=t,
                FoVx=fov,
                FoVy=fov,
                image=img,
                gt_alpha_mask=None,
                image_name="",
                uid=idx
            )

            # 配置渲染器
            raster_settings = GaussianRasterizationSettings(
                image_height=cur_view.image_height,
                image_width=cur_view.image_width,
                tanfovx=tanfov,
                tanfovy=tanfov,
                bg=bg,
                scale_modifier=1.0,
                viewmatrix=cur_view.world_view_transform,
                projmatrix=cur_view.full_proj_transform,
                sh_degree=3,
                campos=cur_view.camera_center,
                prefiltered=False,
                debug=False
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            # 执行渲染
            rendered_image = rasterizer(
                means3D=gs.get_xyz,
                means2D=None,
                shs=gs.get_features,
                colors_precomp=None,
                opacities=gs.get_opacity,
                scales=gs.get_scaling,
                rotations=gs.get_rotation,
                cov3D_precomp=None
            )[0]

            # 记录结果
            syn_imgs.append(rendered_image)
            ref_imgs.append(img.cuda())
            all_labels.append(label)
            gt_masks.append(mask)
            
            # 计算时间
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event)

    return syn_imgs, ref_imgs, all_labels, gt_masks, total_time/len(testset)

def evaluate_anomaly_detection(syn_imgs, ref_imgs, gt_masks, all_labels, args):
    """执行异常检测评估"""
    # 加载预训练模型
    model_name = "resnet"
    if model_name == "effnet":
        with open("PAD_utils/config_effnet.yaml") as f:
            config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        model = ModelHelper(update_config(config).net)
    elif model_name == "resnet":
        import timm
        model = timm.create_model(
            "resnet50",
            pretrained=True, 
            features_only=True,
            out_indices=(1, 2, 3),
            pretrained_cfg_overlay=dict(file='./ckpt/resnet50_a1_0-14fe96d1.pth')
        )
    model.eval()
    model.cuda()

    # 加载RetinexUNet
    light_unet = RetinexUNet().cuda()
    light_unet.load_state_dict(torch.load(f"./ckpt/RetinexUNet_{args.classname}.pth", weights_only=True))
    light_unet.eval()

    # 预处理参数
    CR = torch.nn.MSELoss(reduction='none')
    upscale = v2.Resize(224)
    unet_trans = v2.Compose([
        v2.Resize(448),
    ])
    transform_img = v2.Compose([
        v2.Resize(224),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 计算分数
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    with torch.no_grad():
        # 重构参考图像
        ref_tensors = torch.stack([unet_trans(img) for img in ref_imgs])
        rec_ref = []
        for tensor in ref_tensors:
            rec = light_unet(tensor.unsqueeze(0).cuda()).squeeze()
            rec = white_bg(rec)
            rec_ref.append(rec)
        rec_ref_imgs = torch.stack([upscale(img) for img in rec_ref])
        reconstructed_ref = torch.stack([transform_img(img) for img in rec_ref])
        
        # 处理合成图像
        syn_tensors = torch.stack([transform_img(img) for img in syn_imgs])
        
        # 计算差异
        model_output_ref = model(reconstructed_ref)
        model_output_syn = model(syn_tensors)
        for j in range(len(model_output_ref)):
            var_ref = torch.var(model_output_ref[j])
            var_syn = torch.var(model_output_syn[j])
            print(var_ref, var_syn)
        weights = [0.2, 0.6, 0.2]
        pixel_diff = CR(reconstructed_ref, syn_tensors).sum(dim=1, keepdim=True)
        feature_diff = sum([
            weights[i] * upscale(CR(model(reconstructed_ref)[i], model(syn_tensors)[i]).sum(dim=1, keepdim=True))
            for i in range(len(model(reconstructed_ref)))
        ])
        
        # 融合分数
        combined_diff = pixel_diff + feature_diff
        # combined_diff = pixel_diff
        scores = v2.functional.gaussian_blur(combined_diff, kernel_size=5, sigma=2.0)
        
    # 后处理
    end.record()
    torch.cuda.synchronize()
    cnn_time = start.elapsed_time(end) / len(syn_imgs)
    
    # 归一化分数
    scores_np = scores.squeeze().cpu().numpy()
    scores_min = scores_np.min()
    scores_np = (scores_np - scores_min) / (scores_np.max() - scores_min)
    
    # 计算指标
    gt_mask_np = v2.Resize(224, interpolation=v2.InterpolationMode.NEAREST)(torch.stack(gt_masks)).cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(gt_mask_np.ravel(), scores_np.ravel())
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    fpr, tpr, _ = roc_curve(gt_mask_np.ravel(), scores_np.ravel())
    pixel_rocauc = roc_auc_score(gt_mask_np.ravel(), scores_np.ravel())
    au_pro, au_roc, _, _ = calculate_au_pro_au_roc(gt_mask_np, scores_np)
    
    # 图像级指标
    img_scores = scores_np.reshape(len(syn_imgs), -1).max(axis=1)
    img_labels = np.array([1 if label != 0 else 0 for label in all_labels])
    img_rocauc = roc_auc_score(img_labels, img_scores)
    return (pixel_rocauc, au_pro, au_roc, img_rocauc, cnn_time), scores_np, rec_ref_imgs

import csv
def save_results(args, metrics, result_dir):
    """保存评估结果"""
    content = f"""Pixel_ROCAUC: {metrics[0]:.4f}
            aupro: {metrics[1]:.4f}
            au_roc: {metrics[2]:.4f}
            Image_ROCAUC: {metrics[3]:.4f}
            CNN_time: {metrics[4]:.2f}ms"""
    with open(Path(result_dir)/f'metrics_{args.seed}_{args.trainset_ratio}.txt', 'w') as f:
        f.write(content)
    with open('AD_results.csv', mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # 写入数据行
        writer.writerow([args.classname, metrics[0], metrics[2], metrics[1]])

from torchvision.transforms import Resize
def visualize_results(syn_imgs, ref_imgs, rec_ref, scores, gt_masks, result_dir):
    """生成可视化对比图"""
    comparison_dir = Path(result_dir)/'comparisons'
    comparison_dir.mkdir(exist_ok=True)
    resize = Resize((224, 224))
    for idx in tqdm(range(len(syn_imgs)), desc="Generating visualizations"):
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        # 原始图像
        orig_img = ref_imgs[idx].cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(np.clip(orig_img, 0, 1))
        axes[0].set_title("Original Image")
        axes[0].axis('off')


        retinex_img = rec_ref[idx].cpu().permute(1, 2, 0).numpy()
        axes[1].imshow(np.clip(retinex_img, 0, 1))
        axes[1].set_title("Original Image")
        axes[1].axis('off')
        
        # 重建图像
        recon_img = syn_imgs[idx].cpu().permute(1, 2, 0).numpy()
        axes[2].imshow(np.clip(recon_img, 0, 1))
        axes[2].set_title("Reconstructed Image")
        axes[2].axis('off')
        
        # 异常热图叠加在原始图像上
        heatmap = scores[idx]
        heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap_colored = plt.cm.jet(heatmap_normalized)[..., :3]  # 使用 'jet' 颜色映射，并去掉 alpha 通道
        orig_img_resized = resize(torch.tensor(orig_img).permute(2, 0, 1)).permute(1, 2, 0).numpy()
        overlay_img = np.clip(orig_img_resized, 0, 1) * 0.5 + heatmap_colored * 0.5  # 调整叠加比例
        axes[3].imshow(overlay_img) 
        axes[3].set_title("Anomaly Heatmap (Overlay)")
        axes[3].axis('off')
        
        # 真实掩码
        axes[4].imshow(gt_masks[idx].squeeze().cpu().numpy(), cmap='gray')
        axes[4].set_title("Ground Truth")
        axes[4].axis('off')
        
        plt.tight_layout()
        plt.savefig(comparison_dir/f'comparison_{idx:04d}.png', bbox_inches='tight')
        plt.close()
        if idx == 100:
            break

def main():
    # 初始化
    args = parse_arguments()
    data_path, result_dir = setup_environment(args)
    print("data_path:", data_path)
    total_metrics = {}
    
    # SfM和定位流程
    if not args.skip_loc:
        outputs, image_path, w, h, focal = run_colmap_reconstruction(data_path)
        rec, sfm_time = run_feature_matching(args, outputs, image_path, data_path)
        loc_time = localize_test_images(args, data_path, rec)
        total_metrics.update({"SFM_time": sfm_time, "Loc_time": loc_time})
    
    # 3D高斯训练
    if not args.skip_train:
        train_time = train_gaussian_model(args, data_path, result_dir)
        total_metrics["Train_time"] = train_time
    
    # 图像合成和评估
    syn_imgs, ref_imgs, all_labels, gt_masks, nvs_time = synthesize_reference_images(args, args.data_path, result_dir)
    metrics, scores, rec_ref = evaluate_anomaly_detection(syn_imgs, ref_imgs, gt_masks, all_labels, args)
    total_metrics.update({
        "Pixel_ROCAUC": metrics[0],
        "aupro": metrics[1],
        "au_roc": metrics[2],
        "Image_ROCAUC": metrics[3],
        "NVS_time": nvs_time,
        "CNN_time": metrics[4]
    })
    
    save, visual = True, False
    # 保存和可视化
    if save:
        save_results(args, metrics, result_dir)
    if visual:
        visualize_results(syn_imgs, ref_imgs, rec_ref, scores, gt_masks, result_dir)
    
    # 打印最终结果
    print("\nFinal Metrics:")
    for k, v in total_metrics.items():
        print(f"{k:15}: {v:.4f}" if isinstance(v, float) else f"{k:15}: {v}")

if __name__ == "__main__":
    main()