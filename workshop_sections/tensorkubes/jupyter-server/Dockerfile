FROM gcr.io/tensorflow/tensorflow:latest-devel

RUN apt-get update --fix-missing && apt-get install -y bzip2 \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    curl -L --insecure https://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh -o /anaconda2.sh && \
    /bin/bash /anaconda2.sh -b -p /opt/conda && \
    rm /anaconda2.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install -c conda conda-env

ADD environment.yml environment.yml
RUN conda-env create environment.yml
ADD jupyter_notebook_config.py /root/.jupyter/

ENTRYPOINT ["/run_jupyter.sh"]
