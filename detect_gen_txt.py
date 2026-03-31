import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def detect(save_img=False):
    # 🔔 修改模型路径和类别标签
    names = ['hand_smoke', 'head', 'head_smoke', 'hand', 'call', 'headandhand_smoke']
    source, weights = opt.source, 'yolov7-smoking.pt'  # 强制使用指定模型
    
    # 🔔 移除视频流支持
    webcam = False  # 禁用摄像头/视频流输入
    save_txt = True  # 强制保存txt文件

    # 🔔 修改输出路径结构
    img_folder = Path(source if Path(source).is_dir() else source).parent
    label_dir = img_folder.parent / 'label_txt'
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔔 简化目录创建逻辑
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True) if save_txt else None

    # 初始化设备
    set_logging()
    device = select_device(opt.device)
    
    # 🔔 加载指定模型
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)

    # 数据加载器设置
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 推理流程
    for path, img, im0s, vid_cap in dataset:
        # 图像预处理
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理和NMS
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # 处理检测结果
        for i, det in enumerate(pred):
            p = Path(path)
            # 🔔 修改标签保存路径
            txt_path = str(label_dir / p.stem) + '.txt'
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                # 保存检测结果
                for *xyxy, conf, cls in reversed(det):
                    # 🔔 强制保存YOLO格式
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    with open(txt_path, 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # 格式: class x_center y_center width height

                    # 保持原有绘制逻辑
                    if save_img or opt.view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)

            # 保存带检测结果的图像
            if save_img:
                cv2.imwrite(str(save_dir / p.name), im0s)

if __name__ == '__main__':
    # 🔔 修改参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7-smoking.pt', help='强制使用吸烟检测模型')
    parser.add_argument('--source', type=str, default='inference/images', help='仅支持图片/图片文件夹路径')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='', help='cuda device')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--classes', nargs='+', type=int, help='过滤类别')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--project', default='runs/detect', help='结果保存路径')
    parser.add_argument('--name', default='exp', help='实验结果名称')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖已有结果')
    parser.add_argument('--no-trace', action='store_true', help='禁用模型追踪')
    opt = parser.parse_args()
    
    # 🔔 移除视频相关参数
    opt.update = False  # 禁用模型更新功能
    opt.nosave = False  # 强制保存检测结果
    opt.save_conf = False  # 禁用置信度保存

    with torch.no_grad():
        detect()