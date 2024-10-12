FROM python:3.9.7-slim-buster

RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y \
    vim \
    wget \
    curl \
    git \
    zsh

WORKDIR /sk_cathode
COPY requirements_full.txt .

RUN pip install -r requirements_full.txt
