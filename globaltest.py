import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和预处理器
model_path = '/root/.cache/huggingface/hub/models--nvidia--RADIO/snapshots/3f4562c2f16bbed274414bd88a0d11b84b65a18a'
image_processor = CLIPImageProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)

# 加载图像并转换为张量
def load_images(image_paths):
    images = [pil_to_tensor(Image.open(path).convert("RGB")) for path in image_paths]
    return torch.stack(images).float().div(255.0).to(device)  # 归一化到 [0, 1]

# 定义查询图像和数据库图像路径
query_image_path = 'examples/query3.jpg'
database_image_paths = [
    './examples/db1.jpg',
    './examples/db2.jpg',
    './examples/db3.jpg',
    './examples/db4.jpg',
    './examples/db5.jpg',
]

# 加载所有图像
query_image = load_images([query_image_path])  # 查询图像 (1, C, H, W)
database_images = load_images(database_image_paths)  # 数据库图像 (N, C, H, W)

# 获取全局特征
with torch.no_grad():
    query_features = model(query_image)[0]  # 查询特征 (1, D)
    database_features = model(database_images)[0]  # 数据库特征 (N, D)

# 标准化特征
query_features = torch.nn.functional.normalize(query_features, dim=1)
database_features = torch.nn.functional.normalize(database_features, dim=1)

# 计算距离矩阵 (1, N)
distances = torch.cdist(query_features, database_features, p=2).squeeze(0)  # 移除批维度

# 找到最相似图像的索引
best_match_index = torch.argmin(distances).item()

# 打印结果
print(f"最相似的数据库图像是: db{best_match_index + 1}")
print(f"所有距离（越小越相似）: {distances.cpu().tolist()}")

# 可视化查询图像和最匹配结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(query_image_path))
plt.title("Query Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(Image.open(database_image_paths[best_match_index]))
plt.title(f"Best Match: db{best_match_index + 1}")
plt.axis("off")

plt.show()

