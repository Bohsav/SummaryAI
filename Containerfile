FROM continuumio/miniconda3

RUN mkdir proj
WORKDIR proj

COPY . .

RUN conda install pip
