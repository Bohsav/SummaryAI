FROM mambaorg/micromamba:jammy-cuda-12.2.2

EXPOSE 8889

WORKDIR /home/$MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

RUN micromamba install -y -n base -f environment.yaml && \
    micromamba clean --all --yes

USER $MAMBA_USER

ENTRYPOINT jupyter-lab --no-browser --ip=0.0.0.0 --port=8889

# EXPOSE 8889
#
#
# RUN groupadd -g 1000 jupytergroup && useradd -m -u 1000 -g jupytergroup jupyteruser
# WORKDIR /home/jupyteruser
# COPY . .
#
# RUN conda env create -f environment.yaml
# RUN conda clean -afy

# USER jupyteruser
#
# ENTRYPOINT conda activate SummaryAI && jupyter-lab --no-browser --ip=0.0.0.0
