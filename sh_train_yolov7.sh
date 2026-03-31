
# train p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
  # 吸烟检测
python train.py --workers 8 --device 0 --batch-size 16 --data data/smoking.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7-smoking --hyp data/hyp.scratch.p5.yaml
python train.py --workers 8 --device 0 --batch-size 16 --data data/smoking-cls.yaml --img 640 640 --cfg cfg/training/yolov7-smoking.yaml --weights '' --name yolov7-smoking-scratch --hyp data/hyp.scratch.p5.yaml
  # 烟火检测
python train.py --workers 8 --device 0 --batch-size 16 --data data/fireworks.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7-fireworks --hyp data/hyp.scratch.p5.yaml

# train p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml

# finetune p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
  # 吸烟检测
python train.py --workers 8 --device 0 --batch-size 16 --data data/smoking.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'weights/yolov7_training.pt' --name yolov7-custom-finetune --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 16 --data data/smoking.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'weights/yolov7.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 16 --data data/smoking-cls.yaml --img 640 640 --cfg cfg/training/yolov7-smoking.yaml --weights 'weights/yolov7_training.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 16 --epochs 360 --data data/smoking-cls.yaml --img 640 640 --cfg cfg/training/yolov7-smoking.yaml --weights 'weights/yolov7_training.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.002.yaml

python train.py --workers 8 --device 0 --batch-size 16 --epochs 100 --data data/person-c1.yaml --img 640 640 --cfg cfg/training/yolov7-person-c1.yaml --weights 'weights/yolov7.pt' --name yolov7-person-finetune --hyp data/hyp.scratch.custom-person.yaml
python train.py --workers 8 --device 0 --batch-size 16 --epochs 300 --data data/smoking-c2.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c2.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.smoking.c2.yaml

python train.py --workers 8 --device 0 --batch-size 16 --epochs 270 --data data/smoking-c3.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c3.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.smoking.c3-2.yaml
  # 安全帽检测
python train.py --workers 8 --device 0 --batch-size 16 --epochs 300 --data data/helmet-c2.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c2.yaml --weights 'weights/yolov7.pt' --name yolov7-helmet-ft --hyp data/hyp.scratch.custom.yaml

  # A10
python train.py --workers 8 --device 0 --batch-size 32 --epochs 200 --data data/smoking-cls3.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c3.yaml --weights 'weights/yolov7_training.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 900 --data data/smoking-cls3.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c3.yaml --weights 'weights/yolov7_training.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 600 --data data/smoking-c6.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c6.yaml --weights 'weights/yolov7_training.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.002.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 600 --image-weights --data data/smoking-c6.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c6.yaml --weights 'weights/yolov7_training.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.002.yaml
            # adam
python train.py --workers 8 --device 0 --batch-size 32 --epochs 100 --adam --data data/person-c1.yaml --img 640 640 --cfg cfg/training/yolov7-person-c1.yaml --weights 'weights/yolov7.pt' --name yolov7-person-finetune --hyp data/hyp.scratch.custom-person.yaml
            # sgd
python train.py --workers 8 --device 0 --batch-size 32 --epochs 120 --data data/person-c1.yaml --img 640 640 --cfg cfg/training/yolov7-person-c1.yaml --weights 'weights/yolov7.pt' --name yolov7-person-finetune --hyp data/hyp.scratch.custom-person.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/person-c1.yaml --img 640 640 --cfg cfg/training/yolov7-person-c1.yaml --weights 'weights/yolov7.pt' --name yolov7-person-finetune --hyp data/hyp.scratch.custom-person.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/smoking-c6.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c6.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 303 --data data/smoking-c6.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c6.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 330 --data data/smoking-c6.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c6.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/smoking-c6.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c6.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-finetune --hyp data/hyp.scratch.custom.003.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/ebike-c3.yaml --img 640 640 --cfg cfg/training/yolov7-ebike-c3.yaml --weights 'weights/yolov7.pt' --name yolov7-ebike-finetune --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/ebike-c3.yaml --img 640 640 --cfg cfg/training/yolov7-ebike-c3.yaml --weights 'weights/yolov7.pt' --name yolov7-ebike-finetune --hyp data/hyp.scratch.custom.ebike.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/smoking-c3.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c3.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 360 --data data/smoking-c3.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c3.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.smoking.c3.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/smoking-c2.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c2.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.smoking.c3.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/smoking-c3.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c3.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 360 --data data/smoking-c2.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c2.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.smoking.c2.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 270 --data data/smoking-c3.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c3.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.smoking.c3-2.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 270 --data data/smoking-c3.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c3.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.smoking.c3-2.yaml
python train.py --workers 8 --device 0 --batch-size 16 --epochs 300 --data data/smoking-c3.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c3.yaml --weights 'weights/yolov7.pt' --name yolov7-smoking-ft --hyp data/hyp.scratch.custom.smoking.c3-2.yaml
  # 安全帽检测
python train.py --workers 8 --device 0 --batch-size 32 --epochs 160 --data data/helmet-c2.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c2.yaml --weights 'weights/helmet-20250901-t4-c2-best.pt' --name yolov7-helmet-ft --hyp data/hyp.scratch.custom.helmet.c2.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 260 --data data/helmet-c2.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c2.yaml --weights 'weights/helmet-20250901-t4-c2-best.pt' --name yolov7-helmet-ft --hyp data/hyp.scratch.custom.helmet.c2.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/helmet-c2.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c2.yaml --weights 'weights/yolov7.pt' --name yolov7-helmet-ft --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 32 --epochs 300 --data data/helmet-c2.yaml --img 640 640 --cfg cfg/training/yolov7-smoking-c2.yaml --weights 'weights/helmet-20250901-t4-c2-best.pt' --name yolov7-helmet-ft --hyp data/hyp.scratch.custom.yaml

# finetune p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-custom.yaml --weights 'yolov7-w6_training.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml

python train.py --workers 8 --device 1 \
--batch-size 32 \
--data data/ywj.yaml \
--img 640 640 \
--cfg cfg/training/yolov7-tiny.yaml \
--weights '' \
--name yolov7-tiny \
--hyp data/hyp.scratch.tiny.yaml

# 测试
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
python test.py --data data/smoking.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights weights/smoking_v7_3.9.pt --name yolov7_640_val_smoking

# 推理
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets-test/smoking
python detect.py --weights runs/train/yolov7-smoking-finetune/weights/best.pt --conf 0.25 --img-size 640 --source datasets-test/smoking
  # 保存推理结果为yolo格式的txt文件
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test --save-txt
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-person-detection-2024-0424/train/images --save-txt
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-person-detection-2024-0424/val/images --save-txt
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-infer-txt/images-train-copy-02-part --save-txt
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/HD06--search/smoking_baidu --save-txt
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/HD00--prd-rename --save-txt

python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking/images/val --save-txt --save-conf
python detect.py --weights weights/smoking-t4-20250430.pt --conf 0.25 --img-size 640 --source datasets/smoking/images/val --save-txt --save-conf
python detect.py --weights weights/smoking-a10-20250430.pt --conf 0.25 --img-size 640 --source datasets/smoking/images/val --save-txt --save-conf
python detect.py --weights weights/smoking-t4-20250502.pt --conf 0.25 --img-size 640 --source datasets/smoking/images/val --save-txt --save-conf

python detect.py --weights weights/smoking-c6-def-20250430.pt --conf 0.25 --img-size 640 --source datasets/20250508-162820 --save-txt --save-conf
python detect.py --weights weights/smoking-c6-def-20250430.pt --conf 0.25 --img-size 640 --source datasets/20250512-161641 --save-txt --save-conf
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/20250512-161641 --save-txt --save-conf
python detect.py --weights weights/smoking-c6-hyp2-epo600-20250508.pt --conf 0.25 --img-size 640 --source datasets/smoking/images/val --save-txt --save-conf
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/all-images-jpg-nolabel --save-txt
python detect.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/0515-0604--part--005 --save-txt
python detect.py --weights weights/ebike-v1.pt --conf 0.50 --img-size 640 --source datasets/ebike/val --save-txt

python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.15 --img-size 640 --source datasets/PRD-Frames/smokeDetect --save-txt --save-conf --save-class 0 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.15 --img-size 640 --source datasets/PRD-Frames/002/smokeDetect --save-txt --save-conf --save-class 0 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.15 --img-size 640 --source datasets/PRD-Frames/smokeFireDetect --save-txt --save-conf --save-class 0 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.15 --img-size 640 --source datasets/PRD-Frames/002/smokeFireDetect --save-txt --save-conf --save-class 0 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.30 --img-size 640 --source datasets/PRD-Frames/smoke-model-infer-data --save-txt --save-conf --save-class 0 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.15 --img-size 640 --source datasets/PRD-Frames/003/smokeDetect --save-txt --save-conf --save-class 0 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.15 --img-size 640 --source datasets/PRD-Frames/003/smokeFireDetect --save-txt --save-conf --save-class 0 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.30 --img-size 640 --source datasets/PRD-Frames/smoke-model-infer-data-02 --save-txt --save-conf --save-class 0 2 5
python detect_opt.py --weights weights/smoking-c6-def-20250430.pt --conf 0.86 --img-size 640 --source datasets/shaixuan--20250514 --save-txt --save-conf --save-class 0 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.86 --img-size 640 --source datasets/shaixuan--20250514 --save-txt --save-conf --save-class 0 2 5

python detect_opt.py --weights weights/yolov7-e6e.pt --conf 0.15 --img-size 1280 --source datasets/alarm_person_pro --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7-e6e.pt --conf 0.30 --img-size 1280 --source datasets/alarm_person_pro_model15 --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7-e6e.pt --conf 0.30 --img-size 1280 --source datasets/coco/train2017-person-no --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7-e6e.pt --conf 0.30 --img-size 1280 --source datasets/PRD-Frames/003/smokeDetect --save-txt --save-conf --save-class 0 --classes 0

python detect_opt.py --weights weights/yolov7.pt --conf 0.25 --img-size 640 --source datasets/person-test/images --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7.pt --conf 0.05 --img-size 640 --source datasets/person-test/images --save-txt --save-conf --save-class 0 --classes 0
# python detect_opt.py --weights weights/yolov7.pt --conf 0.05 --img-size 640 --source datasets/person-detection/images/val --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/epoch_049.pt --conf 0.05 --img-size 640 --source datasets/person-test/images --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/epoch_099.pt --conf 0.05 --img-size 640 --source datasets/person-test/images --save-txt --save-conf --save-class 0 --classes 0
# python detect_opt.py --weights weights/epoch_049.pt --conf 0.05 --img-size 640 --source datasets/person-detection/images/val --save-txt --save-conf --save-class 0 --classes 0

python detect_opt.py --weights weights/yolov7.pt --conf 0.25 --img-size 640 --source datasets/smoking-all-images-jpg --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7-e6e.pt --conf 0.25 --img-size 1280 --source datasets/smoking-all-images-jpg --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7.pt --conf 0.25 --img-size 640 --source datasets/step1-all-images-jpg --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7.pt --conf 0.25 --img-size 640 --source datasets/step2-all-images-jpg --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7-e6e.pt --conf 0.25 --img-size 1280 --source datasets/e-bike/images-hq --save-txt --save-conf --save-class 0 1 3 --classes 0 1 3
python detect_opt.py --weights weights/yolov7.pt --conf 0.40 --img-size 640 --source datasets/images-4300 --save-txt --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/666-ori --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/0605-0625 --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/0626-0702 --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7.pt --conf 0.50 --img-size 640 --source datasets/smoking-test/0626-0702 --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/0626-0702-crop --save-txt --save-conf --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking-v1.2.pt --conf 0.50 --img-size 640 --source datasets/smoking-test/0626-0702-crop --save-txt --save-conf --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/0626-0702-crop-sel-rd --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/yolov7.pt --conf 0.50 --img-size 640 --source datasets/smoking-test/0703-0714-rd --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/0703-0714-rd-crop --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/0703-0714-rd-crop-sel --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/step12-crop-images+labels-000+003-images --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/002-sel --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/001-sel --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/003-sel --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/004-sel --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/005-sel --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.60 --img-size 640 --source datasets/smoking-test/003-sel --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/016-01 --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/yolov7.pt --conf 0.60 --img-size 640 --source datasets/smoking-test/image-fullsize--remain --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/zzz--empty-rd --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/image-fullsize--remain-crop --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/image-fullsize--remain-crop --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/train-val-test-data --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/train-val-test-data-hit --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking-v1.2.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/image-fullsize--remain-crop-empty --save-txt --save-class 2 4 5 --classes 2 4 5
python detect_opt.py --weights weights/smoking-v1.2.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/train-val-test-data-cvt --save-txt --save-class 2 4 5 --classes 2 4 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/image-cropsize-004-smoking-remain --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/train-val-test-data-cvt-eff-rename --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/yolov7.pt --conf 0.60 --img-size 640 --source datasets/smoking-test/image-fullsize--prd-0723 --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/005-201 --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/image-fullsize--prd-0723-rd-crop --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/image-fullsize--prd-0723-rd-crop-eff --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking-v1.2.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/crop-002 --save-txt --save-class 2 4 5 --classes 2 4 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/crop-002-eff --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/crop_my_sel --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.10 --img-size 640 --source datasets/smoking-test/crop-001 --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking-v1.2.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/crop-001 --save-txt --save-class 2 4 5 --classes 2 4 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/head-all --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/yolov7.pt --conf 0.50 --img-size 640 --source datasets/smoking-test/prd-0515-0604-oth-rd --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/yolov7.pt --conf 0.50 --img-size 640 --source datasets/smoking-test/prd-0723-0805-rd --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.15 --img-size 640 --source datasets/smoking-test/prd-0515-0604-oth-rd-crop --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking-v1.3.pt   --conf 0.15 --img-size 640 --source datasets/smoking-test/prd-0515-0604-oth-rd-crop --save-txt --save-class 0 2 --classes 0 2
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.15 --img-size 640 --source datasets/smoking-test/prd-0723-0805-rd-crop --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking-v1.3.pt   --conf 0.15 --img-size 640 --source datasets/smoking-test/prd-0723-0805-rd-crop --save-txt --save-class 0 2 --classes 0 2
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/prd-0515-0604-oth-rd-crop-sel --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/prd-0723-0805-rd-crop-sel --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.15 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking-v1.3.pt   --conf 0.15 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 0 2 --classes 0 2
python detect_opt.py --weights weights/yolov7.pt --conf 0.50 --img-size 640 --source datasets/smoking-test/video-frames-sel --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/video-frames-sel-crop --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/yolov7.pt --conf 0.66 --img-size 640 --source datasets/smoking-test/prd-0805-0820-rd --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/prd-0805-0820-rd-crop --save-txt --save-class 1 2 5 --classes 1 2 5
python detect_opt.py --weights weights/yolov7.pt --conf 0.66 --img-size 640 --source datasets/smoking-test/prd-0820-0825-rd --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/prd-0820-0825-rd-crop-sel --save-txt --save-class 1 2 5 --classes 1 2 5

python detect_opt.py --weights weights/smoking_v7_3.9.pt --conf 0.20 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking-v1.1.pt --conf 0.20 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking-v1.2.pt --conf 0.20 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 2 5 --classes 2 5
python detect_opt.py --weights weights/smoking-v1.3.pt --conf 0.20 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 0 2 --classes 0 2
python detect_opt.py --weights weights/smoking-v1.3.2.pt --conf 0.20 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 0 2 --classes 0 2
python detect_opt.py --weights weights/smoking-v1.3.3.pt --conf 0.20 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 0 2 --classes 0 2
python detect_opt.py --weights weights/smoking-20250901-a10-c3-best-e274.pt --conf 0.20 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 0 2 --classes 0 2
python detect_opt.py --weights weights/smoking-20250829-a10-c3-best.pt --conf 0.20 --img-size 640 --source datasets/smoking-test/1 --save-txt --save-class 0 2 --classes 0 2
python detect_opt.py --weights weights/yolov7.pt --conf 0.50 --img-size 640 --source datasets/smoking-test/prd-0826-0903-rd --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking-v1.3.2.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/prd-0826-0903-rd-crop-sel --save-txt
python detect_opt.py --weights weights/yolov7.pt --conf 0.50 --img-size 640 --source datasets/smoking-test/prd-0904-0918-rd --save-txt --save-conf --save-class 0 --classes 0
python detect_opt.py --weights weights/smoking-v1.3.2.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/prd-0904-0918-rd-crop-sel --save-txt
python detect_opt.py --weights weights/smoking-v1.3.2.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/prd-0515-0604-oth-rd-crop-sel-sel --save-txt
python detect_opt.py --weights weights/smoking-v1.3.2.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/prd-0515-0604-oth-rd-crop-sel-02-sel --save-txt

python detect_opt.py --weights weights/helmet-20250901-t4-c2-best.pt --conf 0.25 --img-size 640 --source datasets/smoking-test/helmet-rd --save-txt --save-class 0 1 --classes 0 1
python detect_opt.py --weights weights/helmet-20250901-t4-c2-best.pt --conf 0.40 --img-size 640 --source datasets/coco2017/train2017-person-no --save-txt --save-conf --save-class 0 1 --classes 0 1
python detect_opt.py --weights weights/helmet-20250901-t4-c2-best.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/rd-0905-all-helmet --save-txt --save-class 0 1 --classes 0 1
python detect_opt.py --weights weights/helmet-20250901-t4-c2-best.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/rd-0905-all-person --save-txt --save-class 0 1 --classes 0 1
python detect_opt.py --weights weights/helmet-20250901-t4-c2-best.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/rd-0905-all --save-txt --save-conf --save-class 0 1 --classes 0 1
python detect_opt.py --weights weights/helmet-20250916-a10-c2-best.pt --conf 0.70 --img-size 640 --source datasets/smoking-test/test-image-01 --save-txt --save-class 1 --classes 1
python detect_opt.py --weights weights/helmet-20250916-a10-c2-best.pt --conf 0.70 --img-size 640 --source datasets/smoking-test/rd-0905-all-rd --save-txt --save-class 1 --classes 1
python detect_opt.py --weights weights/helmet-20250916-a10-c2-best.pt --conf 0.70 --img-size 640 --source datasets/smoking-test/helmet-rd-01 --save-txt --save-class 1 --classes 1
python detect_opt.py --weights weights/helmet-20250916-a10-c2-best.pt --conf 0.70 --img-size 640 --source datasets/coco2017/train2017-person-no --save-txt --save-class 1 --classes 1
python detect_opt.py --weights weights/helmet-20250916-a10-c2-best.pt --conf 0.70 --img-size 640 --source datasets/smoking-test/helmet-frames-20250918-rd --save-txt --save-conf --save-class 1 --classes 1
python detect_opt.py --weights weights/helmet-20250917-a10-c2-best.pt --conf 0.70 --img-size 640 --source datasets/smoking-test/helmet-frames-20250918-rd --save-txt --save-conf --save-class 1 --classes 1
python detect_opt.py --weights weights/helmet-20250919-a10-c2-best-rs.pt --conf 0.70 --img-size 640 --source datasets/smoking-test/helmet-frames-20250918-rd --save-txt --save-conf --save-class 1 --classes 1
python detect_opt.py --weights weights/helmet-20250917-a10-c2-best.pt --conf 0.40 --img-size 640 --source datasets/smoking-test/hat-google --save-txt --save-class 0 1 --classes 0 1
python detect_opt.py --weights weights/helmet-20250923-a10-c2-best.pt --conf 0.70 --img-size 640 --source datasets/smoking-test/helmet-frames-20250918-rd --save-txt --save-conf --save-class 1 --classes 1
python detect_opt.py --weights weights/helmet-20250923-a10-c2-best.pt --conf 0.70 --img-size 640 --source datasets/smoking-test/helmet-rd-01 --save-txt --save-conf --save-class 1 --classes 1

python test.py --data data/coco.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights weights/yolov7.pt --name yolov7_640_val --verbose
python test.py --data data/coco.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights weights/yolov7-e6e.pt --name yolov7_640_val --verbose
python test.py --data data/coco.yaml --img 1280 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights weights/yolov7-e6e.pt --name yolov7_640_val --verbose
python test.py --data data/person-c1.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights weights/epoch_099.pt --name yolov7_640_val --verbose
python test.py --data data/person-c80.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights weights/yolov7.pt --name yolov7_640_val --verbose
python test_opt.py --data data/coco.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights weights/yolov7.pt --name yolov7_640_val --verbose




