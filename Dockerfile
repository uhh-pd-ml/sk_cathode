FROM python:3.9.7-slim-buster

RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y \
    vim \
    wget \
    curl \
    git \
    zsh

WORKDIR /sk_cathode
COPY requirements.txt .

# create venv, activate it, and install requirements
RUN apt-get update && apt-get install -y python3-venv
RUN python3 -m venv sk_cathode_venv 
RUN /sk_cathode/sk_cathode_venv/bin/python3 -m pip install --upgrade pip
RUN . sk_cathode_venv/bin/activate 
RUN pip install -r requirements.txt
