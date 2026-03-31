

import os
import shutil
import argparse
from tqdm import tqdm
import time

def classify_images_by_confidence():
    # 配置路径
    path1 = "/root/AI_Service/project/yolov7/yolov7/datasets/coco2017/train2017-person-no"  # 原始图片目录
    path2 = "/root/AI_Service/project/yolov7/yolov7/runs/detect/exp/labels"  # 检测结果目录
    
    # 目标目录配置
    target_dirs = {
        "9": "/root/AI_Service/project/yolov7/yolov7/datasets/coco2017/head-hard-9",
        "8": "/root/AI_Service/project/yolov7/yolov7/datasets/coco2017/head-hard-8",
        "7": "/root/AI_Service/project/yolov7/yolov7/datasets/coco2017/head-hard-7",
        "6": "/root/AI_Service/project/yolov7/yolov7/datasets/coco2017/head-hard-6",
        "5": "/root/AI_Service/project/yolov7/yolov7/datasets/coco2017/head-hard-5",
        "4": "/root/AI_Service/project/yolov7/yolov7/datasets/coco2017/head-hard-4"
    }
    
    # 置信度阈值
    confidence_thresholds = {
        "9": 0.9,
        "8": 0.8,
        "7": 0.7,
        "6": 0.6,
        "5": 0.5
    }
    
    # 支持的图片格式
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp']
    
    print("=" * 80)
    print("图片和检测结果分类工具 - 基于目标检测置信度")
    print("=" * 80)
    print(f"原始图片目录: {path1}")
    print(f"检测结果目录: {path2}")
    print("分类规则:")
    print("1. 最高置信度 >= 0.9 -> head-hard-9")
    print("2. 最高置信度 >= 0.8 且 < 0.9 -> head-hard-8")
    print("3. 最高置信度 >= 0.7 且 < 0.8 -> head-hard-7")
    print("4. 最高置信度 >= 0.6 且 < 0.7 -> head-hard-6")
    print("5. 最高置信度 >= 0.5 且 < 0.6 -> head-hard-5")
    print("6. 最高置信度 < 0.5 -> head-hard-4")
    print("-" * 80)
    
    # 创建所有目标目录
    for dir_path in target_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 检查目录是否存在
    for path in [path1, path2]:
        if not os.path.exists(path):
            print(f"错误: 目录不存在 - {path}")
            return
    
    # 获取所有检测结果文件
    txt_files = [f for f in os.listdir(path2) if f.endswith('.txt')]
    if not txt_files:
        print("检测结果目录中没有找到任何TXT文件")
        return
    
    print(f"找到 {len(txt_files)} 个检测结果文件")
    
    # 统计信息
    files_processed = 0
    pairs_copied = 0
    images_missing = 0
    category_counts = {key: 0 for key in target_dirs.keys()}
    copy_errors = 0
    
    # 处理每个检测结果文件
    for txt_file in tqdm(txt_files, desc="处理文件", unit="file"):
        files_processed += 1
        
        # 获取基础文件名（不含扩展名）
        base_name = os.path.splitext(txt_file)[0]
        
        # 在路径1中查找对应的图片文件
        img_file = None
        for ext in image_exts:
            possible_img = base_name + ext
            if os.path.exists(os.path.join(path1, possible_img)):
                img_file = possible_img
                break
        
        if not img_file:
            images_missing += 1
            continue
        
        # 读取检测结果文件
        txt_path = os.path.join(path2, txt_file)
        max_confidence = 0.0  # 初始化为0
        
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            # 解析每行的置信度
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6:  # 确保有置信度值
                    confidence = float(parts[5])
                    if confidence > max_confidence:
                        max_confidence = confidence
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {str(e)}")
            continue
        
        # 确定目标目录
        target_category = "4"  # 默认类别
        for cat, threshold in confidence_thresholds.items():
            if max_confidence >= threshold:
                target_category = cat
                break
        
        # 更新统计
        category_counts[target_category] += 1
        
        # 复制图片和检测结果到目标目录
        target_dir_path = target_dirs[target_category]
        
        # 复制图片
        src_img_path = os.path.join(path1, img_file)
        dst_img_path = os.path.join(target_dir_path, img_file)
        
        # 复制检测结果
        src_txt_path = os.path.join(path2, txt_file)
        dst_txt_path = os.path.join(target_dir_path, txt_file)
        
        try:
            # 复制图片
            shutil.copy2(src_img_path, dst_img_path)
            
            # 复制检测结果
            shutil.copy2(src_txt_path, dst_txt_path)
            
            pairs_copied += 1
        except Exception as e:
            print(f"复制文件时出错: {str(e)}")
            copy_errors += 1
    
    # 结果报告
    print("\n" + "=" * 80)
    print("处理完成!")
    print(f"处理的检测结果文件数量: {files_processed}")
    print(f"成功复制的图片-检测结果对数量: {pairs_copied}")
    print(f"找不到对应图片的数量: {images_missing}")
    print(f"复制错误数量: {copy_errors}")
    print("-" * 80)
    print("分类统计:")
    for cat, count in category_counts.items():
        print(f"类别 {cat}: {count} 张图片 ({count/files_processed*100:.2f}%)")
    print("-" * 80)
    
    # 各目录文件数量统计
    for cat, dir_path in target_dirs.items():
        files = os.listdir(dir_path)
        images = [f for f in files if any(f.lower().endswith(ext) for ext in image_exts)]
        txts = [f for f in files if f.lower().endswith('.txt')]
        print(f"目录 {dir_path}:")
        print(f"  图片数量: {len(images)}")
        print(f"  检测结果数量: {len(txts)}")
    
    print("=" * 80)

if __name__ == '__main__':
    start_time = time.time()
    classify_images_by_confidence()
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")



