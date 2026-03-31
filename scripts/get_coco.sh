#!/bin/bash
# COCO 2017 dataset http://cocodataset.org
# Download command: bash ./scripts/get_coco.sh

# Download/unzip labels
d='./'  # 解压目录（默認當前目錄）
f='coco2017labels-segments.zip'  # 本地zip文件名
echo "Extracting $f to $d ..."
unzip -q $f -d $d && rm $f  # 直接解压并删除文件（无下载步骤）
echo "Unzip completed. Original zip file removed."

# Download/unzip images
d='./coco/images' # 解压目录
# 直接使用本地已存在的文件名
f1='train2017.zip' # 确保这些文件已存在于当前目录
f2='val2017.zip'
f3='test2017.zip'
for f in $f1 $f2 $f3; do
    # 跳过下载步骤，直接执行解压和删除
    if [ -f "$f" ]; then
        echo "Processing $f ..."
        unzip -q $f -d $d && rm $f &
    else
        echo "Warning: $f not found in current directory"
    fi
done
wait # 等待所有后台任务完成
echo "All files processed"


# # Download/unzip labels
# d='./' # unzip directory
# url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
# f='coco2017labels-segments.zip' # or 'coco2017labels.zip', 68 MB
# echo 'Downloading' $url$f ' ...'
# curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

# # Download/unzip labels
# d='./' # unzip directory
# url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
# f='coco2017labels-segments.zip' # or 'coco2017labels.zip', 68 MB
# echo 'Downloading' $url$f ' ...'
# curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

# # Download/unzip images
# d='./coco/images' # unzip directory
# url=http://images.cocodataset.org/zips/
# f1='train2017.zip' # 19G, 118k images
# f2='val2017.zip'   # 1G, 5k images
# f3='test2017.zip'  # 7G, 41k images (optional)
# for f in $f1 $f2 $f3; do
#   echo 'Downloading' $url$f '...'
#   curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background
# done
# wait # finish background tasks
