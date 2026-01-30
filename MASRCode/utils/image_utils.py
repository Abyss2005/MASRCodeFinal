import os
import base64
import random


def get_image_base64(image_path):
    """将本地图片转换为 Base64 编码字符串"""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def find_entity_images(entity_id, image_root, dataset_name, max_imgs=5):
    """
    根据数据集类型搜索实体的图像文件夹
    WN18RR: n0xxxx
    FB15k:  .m.xxxx (将 /m/ 替换为 .)
    """
    entity_id = str(entity_id).strip()
    target_dir = ""

    if "wn18" in dataset_name.lower():
        # WN18RR 逻辑
        folder_name = entity_id if entity_id.startswith('n') else f"n{entity_id}"
        target_dir = os.path.join(image_root, folder_name)
    else:
        # FB15k 逻辑: /m/01xxx -> .m.01xxx
        folder_name = entity_id.replace('/', '.')
        if folder_name.startswith('.'):
            target_dir = os.path.join(image_root, folder_name)
        else:
            target_dir = os.path.join(image_root, f".{folder_name}")

    selected_paths = []
    if os.path.exists(target_dir):
        all_files = [
            f for f in os.listdir(target_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        # 随机采样，增加多样性
        if len(all_files) > max_imgs:
            chosen = random.sample(all_files, max_imgs)
        else:
            chosen = all_files
        for f in chosen:
            selected_paths.append(os.path.join(target_dir, f))

    return selected_paths