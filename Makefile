all:
	make b2i
	make train
	make noise
	make add_noise
	make b2in

build:
	docker build -t hsiangjenli/ntust:ml_sec_final .

exec:
	docker run -it  --gpus all --rm -v $(PWD):/workspace hsiangjenli/ntust:ml_sec_final bash

motif:
	wget https://raw.githubusercontent.com/boozallen/MOTIF/master/dataset/motif_reports.csv -P data

b2i:
	cd data && python bin_2_img.py --input_file_folder raw --output_file_folder processed/binary2image

train:
	python train.py --train_csv data/blue/train.csv --test_csv data/blue/test.csv --data_dir data/processed/binary2image --model_name blue_team
	python train.py --train_csv data/red/train.csv --test_csv data/red/test.csv --data_dir data/processed/binary2image --model_name red_team

noise:
	# python noise_extract.py --noise_type normalize --image_dir data/processed/binary2image --output_dir data/processed/noise
	python noise_extract.py --noise_type original --image_dir data/processed/binary2image --output_dir data/processed/noise


add_noise:
	# {'maze': 0, 'icedid': 1, 'egregor': 2, 'shamoon': 3,'gandcrab': 4,'azorult': 5}
	@for virus in icedid gandcrab egregor maze shamoon; do \
		for noise_type in original; do \
			for noise_layer in 1 2 3; do \
				echo $$virus $$noise_type $$noise_layer; \
				if [ "$$virus" = "maze" ]; then \
					noise_sample_num=13; \
				elif [ "$$virus" = "icedid" ]; then \
					noise_sample_num=1; \
				elif [ "$$virus" = "egregor" ]; then \
					noise_sample_num=11; \
				elif [ "$$virus" = "shamoon" ]; then \
					noise_sample_num=14; \
				elif [ "$$virus" = "gandcrab" ]; then \
					noise_sample_num=7; \
				elif [ "$$virus" = "azorult" ]; then \
					noise_sample_num=8; \
				fi; \
				cd data; python bin_add_noise.py --vbinary_input_folder raw --vbinary_output_folder processed/add_noise/$$noise_type --noise_input_folder processed/noise/$$noise_type --noise_class $$virus --noise_sample_num $$noise_sample_num --noise_layer $$noise_layer --test_csv blue/test.csv; \
			done \
		done \
	done

b2in:
	@for virus  in icedid gandcrab egregor maze shamoon; do \
		for noise_type in original; do \
			for noise_layer in 1 2 3; do \
				echo $$virus $$noise_type $$noise_layer; \
				cd data; python bin_2_img.py --input_file_folder processed/add_noise/$$noise_type/$$virus"_"$$noise_layer --output_file_folder processed/add_noise_image/$$noise_type/$$virus"_"$$noise_layer; \
			done \
		done \
	done

test:
	python test.py --test_csv data/blue/test.csv --data_dir data/processed/binary2image --model_name blue_team
	@for virus  in icedid gandcrab egregor maze shamoon; do \
		for noise_type in original; do \
			for noise_layer in 1 2 3; do \
				echo $$virus $$noise_type $$noise_layer; \
				python test.py --test_csv data/blue/test.csv --data_dir data/processed/add_noise_image/$$noise_type/$$virus"_"$$noise_layer --model_name blue_team; \
			done \
		done \
	done

test_vis:
	python noise_extract.py \
	--noise_type original \
	--image_dir data/processed/add_noise_image/original/icedid_2 \
	--test_csv data/blue/test.csv \
	--save_noise False \
	--model_name blue_team

	python noise_extract.py \
	--noise_type original \
	--image_dir data/processed/add_noise_image/original/gandcrab_2 \
	--test_csv data/blue/test.csv \
	--save_noise False \
	--model_name blue_team

	python noise_extract.py \
	--noise_type original \
	--image_dir data/processed/binary2image \
	--test_csv data/blue/test.csv \
	--save_noise False \
	--model_name blue_team

	python noise_extract.py \
	--noise_type original \
	--image_dir data/processed/add_noise_image/original/egregor_1 \
	--test_csv data/blue/test.csv \
	--save_noise False \
	--model_name blue_team