from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from torch.nn.functional import interpolate
from einops import rearrange
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_keypoints_and_descriptors(features, image_shape):
    """
    从池化后的特征图中提取关键点和描述符
    features: (L, C) 特征描述符，L 是空间位置数，C 是描述符维度
    image_shape: 原始图像大小 (H, W)
    返回:
        keypoints: (L, 2) 关键点坐标
        descriptors: (L, C) 描述符
    """
    h, w = int(np.sqrt(features.shape[0])), int(np.sqrt(features.shape[0]))
    coords = np.array(np.meshgrid(np.linspace(0, image_shape[1]-1, w),
                                  np.linspace(0, image_shape[0]-1, h))).reshape(2, -1).T
    descriptors = features  # 已经是 (L, C) 格式
    return coords, descriptors

def compute_distance(desc1, desc2):
    """计算描述符之间的欧氏距离矩阵"""
    desc1 = torch.tensor(desc1).to(device)
    desc2 = torch.tensor(desc2).to(device)
    dist = torch.cdist(desc1, desc2, p=2)  # 使用 PyTorch 的 cdist 快速计算
    return dist.cpu().numpy()

def match_with_ratio_test(desc1, desc2, thresh=0.95):
    """基于 Lowe's ratio test 进行匹配"""
    dist = compute_distance(desc1, desc2)
    nearest = np.argpartition(dist, 2, axis=-1)[:, :2]  # 找到两最近邻
    dist_nearest = np.take_along_axis(dist, nearest, axis=-1)
    valid_mask = dist_nearest[:, 0] <= (thresh**2) * dist_nearest[:, 1]
    matches = np.stack([np.where(valid_mask)[0], nearest[valid_mask][:, 0]], axis=1)
    return matches


def plot_matches(img1, kpts1, img2, kpts2, matches, color=(0, 1, 0), dpi=300):
    """
    绘制匹配结果
    img1: 查询图片
    kpts1: 查询图片关键点
    img2: 数据库图片
    kpts2: 数据库图片关键点
    matches: 匹配对
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    combined_img = np.hstack([img1, img2])
    ax.imshow(combined_img)
    ax.axis('off')

    offset = img1.shape[1]

    # 绘制匹配点和连线
    for m in matches:
        pt1 = kpts1[m[0]]
        pt2 = kpts2[m[1]] + np.array([offset, 0])  # 右侧图像偏移量
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, lw=0.5)  # 绘制线
        ax.scatter(pt1[0], pt1[1], color='red', s=1)  # 在查询图像中标记关键点
        ax.scatter(pt2[0], pt2[1], color='blue', s=1)  # 在数据库图像中标记关键点

    plt.show()


image_processor = CLIPImageProcessor.from_pretrained('/root/.cache/huggingface/hub/models--nvidia--RADIO/snapshots/3f4562c2f16bbed274414bd88a0d11b84b65a18a')
model = AutoModel.from_pretrained('/root/.cache/huggingface/hub/models--nvidia--RADIO/snapshots/3f4562c2f16bbed274414bd88a0d11b84b65a18a', trust_remote_code=True)
model.eval().to(device)

# 加载图片
img1 = Image.open('examples/query1.jpg').convert("RGB")
img2 = Image.open('examples/db4.jpg').convert("RGB")
x1 = pil_to_tensor(img1).to(dtype=torch.float32).div(255.0).unsqueeze(0).to(device)
x2 = pil_to_tensor(img2).to(dtype=torch.float32).div(255.0).unsqueeze(0).to(device)

# 调整分辨率
#res1 = model.get_nearest_supported_resolution(*x1.shape[-2:])
#res2 = model.get_nearest_supported_resolution(*x2.shape[-2:])
#x1 = interpolate(x1, res1, mode="bilinear", align_corners=False)
#x2 = interpolate(x2, res2, mode="bilinear", align_corners=False)

# 提取特征
spatial_features1 = model(x1)[1].detach().cpu().numpy()  # (H, W, C)
spatial_features2 = model(x2)[1].detach().cpu().numpy()  # (H, W, C)
print(spatial_features1.shape, spatial_features2.shape)

patch_size = 16  # 假设每个patch的大小是16
spatial_features1 = rearrange(spatial_features1, 'b (h w) d -> b d h w', h=x1.shape[-2] // patch_size, w=x1.shape[-1] // patch_size)
spatial_features2 = rearrange(spatial_features2, 'b (h w) d -> b d h w', h=x2.shape[-2] // patch_size, w=x2.shape[-1] // patch_size)

print(spatial_features1.shape, spatial_features2.shape)

spatial_features1 = F.normalize(torch.tensor(spatial_features1), dim=1).numpy()
spatial_features2 = F.normalize(torch.tensor(spatial_features2), dim=1).numpy()


def apply_pooling(spatial_features, pool_size=4, stride=2):
    """
    对 spatial_features 应用池化操作，减少维度
    spatial_features: (b, d, h, w)
    pool_size: 池化窗口大小
    stride: 池化的步幅
    返回池化后的特征图
    """
    if isinstance(spatial_features, np.ndarray):
        spatial_features = torch.tensor(spatial_features)
    pooled_features = F.max_pool2d(spatial_features, kernel_size=pool_size, stride=stride)
    return pooled_features
pool_size = 4
stride = 2
pooled_features1 = apply_pooling(spatial_features1, pool_size, stride)
pooled_features2 = apply_pooling(spatial_features2, pool_size, stride)

print(pooled_features1.shape, pooled_features2.shape)

# 去掉 batch 维度
pooled_features1 = pooled_features1.squeeze(0)  # (1280, 31, 31)
pooled_features2 = pooled_features2.squeeze(0)

#重排为 (h*w, c)，以便匹配
pooled_features1 = rearrange(pooled_features1, 'c h w -> (h w) c')  # (31*31, 1280)
pooled_features2 = rearrange(pooled_features2, 'c h w -> (h w) c')  # (31*31, 1280)


# 匹配
kpts1, desc1 = extract_keypoints_and_descriptors(pooled_features1.cpu().numpy(), img1.size)
kpts2, desc2 = extract_keypoints_and_descriptors(pooled_features2.cpu().numpy(), img2.size)
matches = match_with_ratio_test(desc1, desc2)

# 可视化
plot_matches(np.asarray(img1), kpts1, np.asarray(img2), kpts2, matches)

print(f"pooled_features1 shape: {pooled_features1.shape}")
print(f"pooled_features2 shape: {pooled_features2.shape}")
print(f"kpts1 shape: {kpts1.shape}, desc1 shape: {desc1.shape}")
print(f"kpts2 shape: {kpts2.shape}, desc2 shape: {desc2.shape}")
print(f"matches shape: {matches.shape}")

print(f"Descriptor stats (image 1): mean={desc1.mean()}, std={desc1.std()}")
print(f"Descriptor stats (image 2): mean={desc2.mean()}, std={desc2.std()}")





