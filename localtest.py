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
import cv2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_to_match(img1, img2):
    """
    将 img2 调整为与 img1 相同分辨率。
    img1: 基准图像（目标分辨率）
    img2: 待调整图像
    """
    h1, w1 = img1.size[::-1]  # PIL.Image 的 size 是 (宽, 高)，需要反转为 (高, 宽)

    img2_resized = cv2.resize(np.array(img2), (w1, h1), interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(img2_resized)


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

def match_with_ratio_test(desc1, desc2, thresh=0.985):
    """基于 Lowe's ratio test 进行匹配"""
    dist = compute_distance(desc1, desc2)
    nearest = np.argpartition(dist, 2, axis=-1)[:, :2]  # 找到两最近邻
    dist_nearest = np.take_along_axis(dist, nearest, axis=-1)
    valid_mask = dist_nearest[:, 0] <= (thresh**2) * dist_nearest[:, 1]
    matches = np.stack([np.where(valid_mask)[0], nearest[valid_mask][:, 0]], axis=1)
    return matches



def filter_matches_with_ransac(kpts1, kpts2, matches, ransac_thresh=70.0):
    """
    使用 RANSAC 过滤匹配点，估计单应性矩阵并剔除外点。

    参数：
        kpts1: (N, 2) 查询图像中的关键点坐标
        kpts2: (N, 2) 数据库图像中的关键点坐标
        matches: (M, 2) 匹配对
        ransac_thresh: RANSAC 的阈值，单位为像素

    返回：
       H: 单应性矩阵
        filtered_matches: 通过 RANSAC 筛选的匹配对
        mask: RANSAC 的内点掩码
    """
    # 提取匹配点的坐标
    src_pts = np.float32([kpts1[m[0]] for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m[1]] for m in matches]).reshape(-1, 1, 2)

    # 使用 RANSAC 估计单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    # 仅保留内点
    matches_mask = mask.ravel().tolist()
    filtered_matches = [matches[i] for i, valid in enumerate(matches_mask) if valid]

    print(f"Initial matches: {len(matches)}, Filtered matches: {len(filtered_matches)}")
    return H, filtered_matches, matches_mask

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
        ax.scatter(pt1[0], pt1[1], color='red', s=10)  # 在查询图像中标记关键点
        ax.scatter(pt2[0], pt2[1], color='blue', s=10)  # 在数据库图像中标记关键点

    plt.show()

def warp_image(img, H, target_shape):
    """
    使用单应性矩阵 H 对图像进行透视变换
    img: 待变换的图像
    H: 单应性矩阵
    target_shape: 目标图像的尺寸 (width, height)
    """
    # 透视变换，目标图像的尺寸由 target_shape 提供
    warped_img = cv2.warpPerspective(np.array(img), H, (target_shape[0], target_shape[1]))
    return Image.fromarray(warped_img)




def plot_images_side_by_side(img1, img2, title1='Image 1', title2='Image 2'):
    """
    将两张图像并排显示
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1)
    ax[0].set_title(title1)
    ax[0].axis('off')
    ax[1].imshow(img2)
    ax[1].set_title(title2)
    ax[1].axis('off')
    plt.show()


image_processor = CLIPImageProcessor.from_pretrained('/root/autodl-tmp/huggingface/hub/models--nvidia--RADIO/snapshots/3f4562c2f16bbed274414bd88a0d11b84b65a18a')
model = AutoModel.from_pretrained('/root/autodl-tmp/huggingface/hub/models--nvidia--RADIO/snapshots/3f4562c2f16bbed274414bd88a0d11b84b65a18a', trust_remote_code=True)
model.eval().to(device)

# 加载图片
img1 = Image.open('examples/@map@0038.jpg').convert("RGB")
img2 = Image.open('examples/warped_0038.png').convert("RGB")
print(img1.size, img2.size)
img2= resize_to_match(img1, img2)
print(img1.size, img2.size)
x1 = pil_to_tensor(img1).to(dtype=torch.float32).div(255.0).unsqueeze(0).to(device)
x2 = pil_to_tensor(img2).to(dtype=torch.float32).div(255.0).unsqueeze(0).to(device)
print(x1.shape, x2.shape)

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



spatial_features1 = torch.tensor(spatial_features1)
spatial_features2 = torch.tensor(spatial_features2)
spatial_features1= spatial_features1.squeeze(0)  # (1280, 31, 31)
spatial_features2 = spatial_features2.squeeze(0)

print(spatial_features1.shape, spatial_features2.shape)
#重排为 (h*w, c)，以便匹配
pooled_features1 = rearrange(spatial_features1, 'c h w -> (h w) c')  # (31*31, 1280)
pooled_features2 = rearrange(spatial_features2, 'c h w -> (h w) c')  # (31*31, 1280)
print(pooled_features1.shape, pooled_features2.shape)
# 匹配
kpts1, desc1 = extract_keypoints_and_descriptors(pooled_features1.cpu().numpy(), img1.size)
kpts2, desc2 = extract_keypoints_and_descriptors(pooled_features2.cpu().numpy(), img2.size)
matches = match_with_ratio_test(desc1, desc2)

# 使用 RANSAC 过滤匹配点
H, filtered_matches, matches_mask = filter_matches_with_ransac(kpts1, kpts2, matches)

# 可视化过滤后的匹配

plot_matches(np.asarray(img1), kpts1, np.asarray(img2), kpts2, filtered_matches)


print(f"pooled_features1 shape: {pooled_features1.shape}")
print(f"pooled_features2 shape: {pooled_features2.shape}")
print(f"kpts1 shape: {kpts1.shape}, desc1 shape: {desc1.shape}")
print(f"kpts2 shape: {kpts2.shape}, desc2 shape: {desc2.shape}")
print(f"matches shape: {matches.shape}")

print(f"Descriptor stats (image 1): mean={desc1.mean()}, std={desc1.std()}")
print(f"Descriptor stats (image 2): mean={desc2.mean()}, std={desc2.std()}")



# 将 img2 透视变换为与 img1 对齐
warped_img1 = warp_image(img1, H, img2.size)  # 使用 img2.size

# 将两张图像并排显示进行对比
plot_images_side_by_side(img2, warped_img1, title1="Reference Image", title2="Warped Image")





