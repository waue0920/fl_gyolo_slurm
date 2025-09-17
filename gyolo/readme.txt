## /home/waue0920/fl_gyolo_slurm/gyolo/readme.txt
##################################################
to 翃碩：

## 切換
$ conda activate gyolo;

## 路徑
$ cd ~/fl_gyolo_slurm/gyolo;

## 執行
$ python caption/train.py --device 0,1 --batch 8 --epochs 30 --workers 8 \
--data data/coco.yaml --img 640 --cfg models/caption/gyolo.yaml \
--name gyolo-b128 --hyp data/hyps/hyp.scratch-cap.yaml \
--optimizer AdamW --flat-cos-lr --no-overlap --close-mosaic 2 --save-period 1 --noplots \
--weights '../gyolo.pt' \
--project "haha_20250909"
