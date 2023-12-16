exec:
	docker run -it  --gpus all --rm -v $(PWD):/workspace pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime bash

motif:
	wget https://raw.githubusercontent.com/boozallen/MOTIF/master/dataset/motif_reports.csv -P data