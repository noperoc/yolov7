import os

# 配置标签文件夹路径
# label_dir = "datasets/smoking/labels/train"  # 检查训练集和验证集所有标签
label_dir = "datasets/smoking/labels/val"  # 检查训练集和验证集所有标签

max_class_id = 0
invalid_files = []

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue
    with open(os.path.join(label_dir, label_file), "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # 跳过空行或无效行
            class_id = int(parts[0])
            if class_id >= 6:  # 根据 nc=6 检查
                invalid_files.append((label_file, class_id))
            max_class_id = max(max_class_id, class_id)

print(f"最大类别编号: {max_class_id}")
if invalid_files:
    print("发现无效的类别编号：")
    for file, cls_id in invalid_files:
        print(f"文件: {file}, 无效类别: {cls_id}")
else:
    print("所有标签文件的类别编号均有效（0-5）。")
