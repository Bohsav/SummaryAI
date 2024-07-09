FROM continuumio/miniconda3

EXPOSE 8888

RUN groupadd -g 1000 jupytergroup && useradd -m -u 1000 -g jupytergroup jupyteruser
WORKDIR /home/jupyteruser
COPY . .

RUN conda env create -f environment.yaml
RUN conda clean -afy

USER jupyteruser

ENTRYPOINT conda activate SummaryAI && jupyter-lab --no-browser --ip=0.0.0.0
