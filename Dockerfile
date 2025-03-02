FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /docker

COPY . .

RUN pip install . -v

CMD ["python3", "-u", "scripts/sample_train.py"]
 