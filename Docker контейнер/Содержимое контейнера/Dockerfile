FROM python:3.8 AS builder
COPY requirements.txt .
COPY ratings.csv .
RUN apt-get update -y



COPY . /dock
WORKDIR /dock

RUN pip install --user -r requirements.txt


CMD [ "python", "nn.py" ]
