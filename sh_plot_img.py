

import os
import cv2

# 数据集配置清单
DATASETS_CONFIG = [
    {   # 吸烟检测数据集
        "name": "smoking",
        "class_names": ['hand_smoke', 'head', 'head_smoke', 'hand', 'call', 'headandhand_smoke'],
        "colors": [
            (0, 0, 255),    # 红
            (255, 0, 0),    # 蓝
            (0, 255, 0),    # 绿
            (0, 255, 255),  # 黄
            (128, 0, 128),  # 紫
            (0, 165, 255)   # 橙
        ]
    },
    {   # 烟火检测数据集
        "name": "fireworks",
        "class_names": ['fire', 'smoke'],
        "colors": [
            (0, 0, 255),    # 红色表示火焰
            (192, 192, 192) # 灰色表示烟雾
        ]
    }
]

# 全局参数
BASE_PROJECT_DIR = '/root/AI_Service/project/yolov7/yolov7/datasets'
SUBSETS = ['train', 'val']
FONT_SCALE = 0.6
TEXT_THICKNESS = 2
BOX_THICKNESS = 2

def visualize_dataset(dataset_config):
    """处理单个数据集的可视化"""
    dataset_name = dataset_config["name"]
    class_names = dataset_config["class_names"]
    colors = dataset_config["colors"]
    
    dataset_dir = os.path.join(BASE_PROJECT_DIR, dataset_name)
    print(f"\n正在处理数据集: {dataset_name} -> {dataset_dir}")

    for subset in SUBSETS:
        # 构建路径
        image_dir = os.path.join(dataset_dir, 'images', subset)
        label_dir = os.path.join(dataset_dir, 'labels', subset)
        output_dir = os.path.join(dataset_dir, 'imgplot', subset)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 遍历所有图片文件
        for filename in os.listdir(image_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            image_path = os.path.join(image_dir, filename)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)
            
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"错误：无法读取图片 {image_path}")
                continue
            
            # 处理标签
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                img_height, img_width = img.shape[:2]
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    
                    # 验证标签格式
                    if len(parts) != 5:
                        print(f"格式错误：{label_path} 行内容 '{line}'")
                        continue
                    
                    # 解析数据
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        w = float(parts[3]) * img_width
                        h = float(parts[4]) * img_height
                    except (ValueError, IndexError) as e:
                        print(f"数据错误：{label_path} ({e}) 行内容 '{line}'")
                        continue
                    
                    # 验证类别ID合法性
                    if not (0 <= class_id < len(class_names)):
                        print(f"非法类别ID：{class_id} in {label_path}")
                        continue
                    
                    # 计算坐标
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    # 绘制边界框和文字
                    color = colors[class_id]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                    
                    # 添加类别标签
                    text = class_names[class_id]
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS
                    )
                    # 文字背景框
                    cv2.rectangle(img, 
                                (x1, y1 - text_height - 5),
                                (x1 + text_width, y1),
                                color, -1)  # -1表示填充
                    # 文字内容
                    cv2.putText(img, text, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                              (255, 255, 255), TEXT_THICKNESS)  # 白色文字
                    
            # 保存结果
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)
            print(f"生成：{output_path}")

if __name__ == "__main__":
    # for config in DATASETS_CONFIG:
    #     visualize_dataset(config)
    visualize_dataset(DATASETS_CONFIG[1])
    print("\n所有数据集处理完成！输出目录结构：")
    print("├── smoking/imgplot/[train|val]")
    print("└── fireworks/imgplot/[train|val]")


# import os
# import cv2

# # 配置参数
# base_dir = '/root/AI_Service/project/yolov7/yolov7/datasets/smoking'
# subsets = ['train', 'val']  # 处理训练集和验证集
# class_names = ['hand_smoke', 'head', 'head_smoke', 'hand', 'call', 'headandhand_smoke']
# colors = [
#     (0, 0, 255),    # 红 (0)
#     (255, 0, 0),    # 蓝 (1)
#     (0, 255, 0),    # 绿 (2)
#     (0, 255, 255),  # 黄 (3)
#     (128, 0, 128),  # 紫 (4)
#     (0, 165, 255)   # 橙 (5)
# ]
# font_scale = 0.6    # 字体大小系数
# text_thickness = 2   # 文字线宽
# box_thickness = 2    # 框线宽

# for subset in subsets:
#     # 构建路径
#     image_dir = os.path.join(base_dir, 'images', subset)
#     label_dir = os.path.join(base_dir, 'labels', subset)
#     output_dir = os.path.join(base_dir, 'imgplot', subset)
    
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 遍历图片目录
#     for filename in os.listdir(image_dir):
#         # 检查图片格式
#         if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#             continue
        
#         image_path = os.path.join(image_dir, filename)
#         label_filename = os.path.splitext(filename)[0] + '.txt'
#         label_path = os.path.join(label_dir, label_filename)
        
#         # 读取图片
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"错误：无法读取图片 {image_path}")
#             continue
        
#         # 处理标签文件
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 lines = f.readlines()
            
#             img_height, img_width = img.shape[:2]
            
#             for line in lines:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 parts = line.split()
                
#                 # 校验标签格式
#                 if len(parts) != 5:
#                     print(f"警告：{label_path} 中存在无效行 '{line}'，已跳过")
#                     continue
                
#                 # 解析数据
#                 try:
#                     class_id = int(parts[0])
#                     x_center = float(parts[1]) * img_width
#                     y_center = float(parts[2]) * img_height
#                     w = float(parts[3]) * img_width
#                     h = float(parts[4]) * img_height
#                 except (ValueError, IndexError) as e:
#                     print(f"错误：{label_path} 解析失败 ({e})，行内容：{line}")
#                     continue
                
#                 # 校验类别合法性
#                 if not (0 <= class_id < len(class_names)):
#                     print(f"警告：检测到非法类别ID {class_id}，文件 {label_path}")
#                     continue
                
#                 # 计算坐标
#                 x1 = int(x_center - w/2)
#                 y1 = int(y_center - h/2)
#                 x2 = int(x_center + w/2)
#                 y2 = int(y_center + h/2)
                
#                 # 绘制边界框
#                 color = colors[class_id]
#                 cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)
                
#                 # 添加类别标签
#                 text = class_names[class_id]
#                 (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
#                 cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
#                 cv2.putText(img, text, (x1, y1 - 5), 
#                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
#                            (255, 255, 255), text_thickness)  # 白字黑边更清晰
        
#         # 保存结果
#         output_path = os.path.join(output_dir, filename)
#         cv2.imwrite(output_path, img)
#         print(f"生成：{output_path}")

# print("可视化标注完成！所有图片已保存至 imgplot 目录")