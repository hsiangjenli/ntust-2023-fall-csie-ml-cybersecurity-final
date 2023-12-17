FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY requirements.txt /workspace/requirements.txt

RUN apt-get update -y && apt-get install g++ make git -y
RUN pip install -r /workspace/requirements.txt