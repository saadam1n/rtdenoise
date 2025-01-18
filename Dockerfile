FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /docker

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install . -v

CMD ["python3", "scripts/sample_train.py"]
 