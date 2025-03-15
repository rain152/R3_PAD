import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2
from PIL import Image
from gaussian_splatting.train import *
from gaussian_splatting.scene.cameras import Camera
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# needed for PAD code
from utils_pose_est import DefectDataset
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer 
import torch.nn.functional as F
from Retinex_UNet import RetinexUNet
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from argparse import ArgumentParser

# 定义不同的输入尺寸
input_sizes = [224, 448, 800]
input_size = 448
num_epochs = 50
batch_size = 4


def syn_pseudo_ref_imgs_2(cur_class, model_dir, data_dir):
    trainset = DefectDataset(data_dir, cur_class, "train", True, True)
    train_imgs = torch.cat([a[0][None,...] for a in trainset], dim=0)
    train_poses = np.concatenate([np.array(a["transform_matrix"])[None,...] for a in trainset.camera_transforms["frames"]])
    train_imgs = torch.movedim(torch.nn.functional.interpolate(train_imgs, (800, 800)), 1, 3).numpy()
    fov, tanfov = trainset.camera_angle, np.tan(trainset.camera_angle/2)
    bg = torch.tensor([1.0,1.0,1.0], device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ref_imgs, syn_imgs, times = [], [], 0

    with torch.no_grad():
        gs = GaussianModel(3)
        gs.load_ply(os.path.join(model_dir, f"point_cloud/iteration_{lego_args.iters}/point_cloud.ply"))
        means3D, shs, scales, rotations, opacities = gs.get_xyz, gs.get_features, gs.get_scaling, gs.get_rotation, gs.get_opacity
        for i in tqdm(range(len(trainset)), desc='Rendering test views'):
            start.record()
            img = torch.from_numpy(train_imgs[i])
            img = img.permute(2, 0, 1)
            ref_imgs.append(img)
            
            c2w_init = train_poses[i]
            c2w_init[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w_init)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            c2w_init[:3,:3] = R
            c2w_init[:3,3] = T
            c2w_init = torch.from_numpy(c2w_init).type(torch.float).to("cuda")
            
            cur_view = Camera(1, R=c2w_init[:3,:3].cpu().numpy(), T=c2w_init[:3,3].cpu().numpy(), FoVx=fov, FoVy=fov, image=img, gt_alpha_mask=None, image_name="", uid=1)
            raster_settings = GaussianRasterizationSettings(    # Set up rasterization configuration
                image_height=cur_view.image_height,
                image_width=cur_view.image_width,
                tanfovx=tanfov,
                tanfovy=tanfov,
                bg=bg,
                scale_modifier=1,
                viewmatrix=cur_view.world_view_transform,
                projmatrix=cur_view.full_proj_transform,
                sh_degree=3,
                campos=cur_view.camera_center,
                prefiltered=False,
                debug=False
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            syn_imgs.append(rasterizer(means3D, None, opacities, shs, None, scales, rotations, None)[0])
            end.record()
            torch.cuda.synchronize()
            times += start.elapsed_time(end)
    return syn_imgs, ref_imgs, times/len(trainset)

# 定义一个函数来保存可视化的重构效果
def save_visualization(model, dataloader, save_dir, num_images=100):
    model.eval()  # 设置模型为评估模式
    idx = 0
    with torch.no_grad():  # 禁用梯度计算
        for batch_syn, batch_ref in dataloader:
            batch_syn, batch_ref = batch_syn.cuda(), batch_ref.cuda()
            reconstructed = model(batch_syn)

            batch_syn = batch_syn.cpu().numpy()
            batch_ref = batch_ref.cpu().numpy()
            reconstructed = reconstructed.cpu().numpy()

            # 选择前 num_images 个图像进行可视化
            for i in range(len(batch_syn)):
                original_img = batch_syn[i].transpose(1, 2, 0)
                reconstructed_img = reconstructed[i].transpose(1, 2, 0)
                reference_img = batch_ref[i].transpose(1, 2, 0)

                # 将图像数据裁剪到 [0, 1] 范围内
                original_img = np.clip(original_img, 0, 1)
                reconstructed_img = np.clip(reconstructed_img, 0, 1)
                reference_img = np.clip(reference_img, 0, 1)

                # 绘制图像
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(original_img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(reconstructed_img)
                axes[1].set_title('Reconstructed Image')
                axes[1].axis('off')

                axes[2].imshow(reference_img)
                axes[2].set_title('Reference Image')
                axes[2].axis('off')

                # 保存图像
                save_path = f"{save_dir}/reconstruction_{idx}.png"
                plt.savefig(save_path)
                plt.close(fig)  # 关闭图像以释放内存
                idx += 1
            if idx > num_images:
                break

classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

### prepare args
pre_parser = ArgumentParser(description="Parameters of the LEGO training run")
pre_parser.add_argument("-c", "-classname", metavar="c", type=str, help="current class to run experiments on", default="07Owl")
pre_parser.add_argument("-seed", type=int, help="seed for random behavior", default=0)
pre_parser.add_argument("-iters", type=int, help="number of training iterations for 3DGS", default=15000)
pre_parser.add_argument("-skip_loc", help='skip localization', action='store_true')                 
pre_parser.add_argument("-skip_train", default=True, help='skip training 3dgs')                    
pre_parser.add_argument("-data_path", type=str, help="preprocessed dataset path", default="MAD-Sim/")         
pre_parser.add_argument("-n_match", type=int, default=15, help="num of matches for netvlad image retrieval")   
pre_parser.add_argument("-trainset_ratio", type=float, default=1.0, help="percentage of training samples")     
pre_parser.add_argument("-feature_ext", type=str, default='superpoint', help="feature extractor", choices=["superpoint", "aliked"])                  

lego_args = pre_parser.parse_args()
data_path = os.path.join(lego_args.data_path, lego_args.c)
result_dir = os.path.join(f"results", lego_args.c)
print("saving model to: ", result_dir)
os.makedirs(result_dir, exist_ok=True)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
num_epochs = 100
batch_size = 4

syn_imgs, ref_imgs, nvs_time = syn_pseudo_ref_imgs_2(lego_args.c, result_dir, lego_args.data_path)
tf_img = v2.Compose([
    v2.Resize(input_size),
])
ref = tf_img(torch.stack(ref_imgs))
syn = tf_img(torch.stack(syn_imgs))

# 创建数据集和数据加载器
dataset = TensorDataset(ref, syn)
# indices = np.arange(len(dataset))
# half_indices = indices[::2]  # 每隔一个样本取一个
# sampler = SubsetRandomSampler(half_indices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,)

# 定义感知损失
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg16(pretrained=True).features[:16]  # 使用VGG16的前16层
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return F.l1_loss(x_vgg, y_vgg)

def train_model(model, dataloader, optimizer, scheduler, num_epochs, mse_loss_fn, perceptual_loss_fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    patience = 5
    best_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_syn, batch_ref in dataloader:
            batch_syn, batch_ref = batch_syn.cuda(), batch_ref.cuda()
            reconstructed = model(batch_syn)
            reconstruction_loss = 0.9 * mse_loss_fn(reconstructed, batch_ref) + 0.1 * perceptual_loss_fn(reconstructed, batch_ref)
            loss = reconstruction_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

        if epoch_loss/len(dataloader) < best_loss:
            best_loss = epoch_loss/len(dataloader)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping!")
                break

    end.record()
    print(f"Training completed. Training costs {start.elapsed_time(end)}ms")
    return model

# 在主程序中调用训练函数
model = RetinexUNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
mse_loss_fn = torch.nn.MSELoss()
perceptual_loss_fn = PerceptualLoss().cuda()

train = True
if train:
    model = train_model(model, dataloader, optimizer, scheduler, num_epochs, mse_loss_fn, perceptual_loss_fn)
    model_save_path = os.path.join(f"./ckpt/RetinexUNet_{lego_args.c}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
else:
    print("Skip training!")

model.load_state_dict(torch.load(f"./ckpt/RetinexUNet_{lego_args.c}.pth", weights_only=True))
model.eval()
# 创建一个新的数据加载器，用于可视化
visualize_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# 保存可视化结果
save_visualization(model, visualize_dataloader, save_dir='visualizations')

