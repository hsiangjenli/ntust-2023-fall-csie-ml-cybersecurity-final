exec:
	docker run -it  --gpus all --rm -v $(PWD):/workspace pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime bash

motif:
	wget https://raw.githubusercontent.com/boozallen/MOTIF/master/dataset/motif_reports.csv -P data

b2i:
	cd data && python bin2img.py --input_file_folder raw --output_file_folder binary2image

train:
	python train.py --train_csv data/blue/train.csv --test_csv data/blue/test.csv --data_dir data/binary2image --model_name blue_team
	python train.py --train_csv data/red/train.csv --test_csv data/red/test.csv --data_dir data/binary2image --model_name red_team