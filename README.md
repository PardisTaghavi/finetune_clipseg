a code for fine-tuning clip seg


fine-tuning on cityscapes:
```
python finetune_clipseg.py --data_dir /media/avalocal/T7/pardis/pardis/perception_system/datasets/train_few_shot_leftover --output_dir ./clipseg_finetuned --num_epochs 20 --batch_size 8 --learning_rate 1e-5
```
