
FROM python:3.7-slim

RUN apt-get update && apt-get install -y \
    gcc \
    graphviz

ARG PROJECT_DIR=/usr/src

ENV PROJECT_DIR=${PROJECT_DIR}

COPY . ${PROJECT_DIR}

RUN pip install --user --no-cache-dir -r ${PROJECT_DIR}/requirements.txt
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/bin:$PYTHONPATH

WORKDIR ${PROJECT_DIR}
