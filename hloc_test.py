import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from torchvision.transforms.functional import pil_to_tensor
from einops import rearrange
import matplotlib.pyplot as plt
import torch.nn.functional as F


# **设备设置**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# **模型与预处理器加载**
model_path = "/root/.cache/huggingface/hub/models--nvidia--RADIO/snapshots/3f4562c2f16bbed274414bd88a0d11b84b65a18a"
image_processor = CLIPImageProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)


# **加载并预处理图像**
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = pil_to_tensor(img).float().div(255.0).unsqueeze(0).to(device)  # 归一化到[0,1]
    return img, tensor


# **提取全局描述符**
def extract_global_descriptor(image_tensor):
    """提取图像的全局描述符"""
    with torch.no_grad():
        global_descriptor = model(image_tensor)[0]  # (1, C)
        global_descriptor=torch.nn.functional.normalize(global_descriptor, dim=1)
    return global_descriptor.to(device)  # 保持张量格式，方便距离计算





# **提取局部特征**
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


def plot_matches(img1, kpts1, img2, kpts2, matches, color=(0, 1, 0), dpi=50):
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


# **主逻辑**
if __name__ == "__main__":
    # **加载查询图像**
    query_img, query_tensor = load_image("examples/query3.jpg")

    # **加载数据库图像**
    database_paths = ["./examples/db1.jpg", "./examples/db2.jpg", "./examples/db3.jpg", "./examples/db4.jpg", "./examples/db5.jpg"]
    database_images = [load_image(path) for path in database_paths]

    # **全局匹配**
    query_global = extract_global_descriptor(query_tensor).unsqueeze(0)  # (1, C)
    database_globals = torch.stack([extract_global_descriptor(tensor) for _, tensor in database_images])  # (N, C)

    distances = torch.cdist(query_global, database_globals, p=2).squeeze(0)  # 计算欧氏距离
    best_match_idx = torch.argmin(distances).item()
    best_img, best_tensor = database_images[best_match_idx]

    print(f"Best match: {database_paths[best_match_idx]} with distance: {distances[best_match_idx].item():.4f}")


    # **局部特征点匹配**
    with torch.no_grad():
        query_features = model(query_tensor)[1].cpu().numpy()  # 局部特征
        best_features = model(best_tensor)[1].cpu().numpy()  # 局部特征

    x1 = pil_to_tensor(best_img).to(dtype=torch.float32).div(255.0).unsqueeze(0).to(device)
    x2 = pil_to_tensor(query_img).to(dtype=torch.float32).div(255.0).unsqueeze(0).to(device)
    patch_size = 16  # 假设每个patch的大小是16
    query_features = rearrange(query_features, 'b (h w) d -> b d h w', h=x1.shape[-2] // patch_size,
                                  w=x1.shape[-1] // patch_size)
    best_features= rearrange(best_features , 'b (h w) d -> b d h w', h=x2.shape[-2] // patch_size,
                                  w=x2.shape[-1] // patch_size)


    query_features = F.normalize(torch.tensor(query_features), dim=1).numpy()
    best_features = F.normalize(torch.tensor(best_features), dim=1).numpy()

    query_features = torch.tensor(query_features)
    best_features= torch.tensor(best_features)
    query_features = query_features.squeeze(0)  # (1280, 31, 31)
    best_features = best_features.squeeze(0)

    # 重排为 (h*w, c)，以便匹配
    pooled_features1 = rearrange(query_features, 'c h w -> (h w) c')  # (31*31, 1280)
    pooled_features2 = rearrange(best_features, 'c h w -> (h w) c')

    kpts1, desc1 = extract_keypoints_and_descriptors(pooled_features1.cpu().numpy(), best_img.size)
    kpts2, desc2 = extract_keypoints_and_descriptors(pooled_features2.cpu().numpy(), query_img.size)
    matches = match_with_ratio_test(desc1, desc2)

    # 可视化
    plot_matches(np.asarray(best_img), kpts1, np.asarray(query_img), kpts2, matches)
