FROM tensorflow/tensorflow:2.7.0

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

COPY . /home/QUIP
WORKDIR /home/QUIP

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-distutils python3.8-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    tensorflow==2.7.0 \
    tensorflow-quantum==0.7.2 \
    scikit-learn && \
    pip install --no-cache-dir -r requirements.txt

RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*