FROM python:3.10.14

LABEL key="LeeSungChoi"

ENV PYTHONUNBUFFERED 1

WORKDIR /app

EXPOSE 8000

ARG DEV=false
ARG CMAKE_VERSION=3.30.1
ARG NUM_JOBS=8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        autotools-dev \
        libicu-dev \
        libbz2-dev \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        libssl-dev \
        wget \
        git \
        curl \
        vim \
        gdb \
        valgrind \
        swig \
        zlib1g \
        zlib1g-dev \
        libxml2-dev \
        libeigen3-dev \
        libcairo2-dev \
        libboost-all-dev && \
    apt-get clean

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar xzf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap && \
    make -j${NUM_JOBS} && \
    make install && \
    rm -rf /tmp/*

# Install Open Babel
RUN git clone https://github.com/openbabel/openbabel.git && \
    cd openbabel && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j${NUM_JOBS} && \
    make install

COPY ./requirements.txt /tmp/requirements.txt
COPY ./requirements.dev.txt /tmp/requirements.dev.txt

RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /tmp/requirements.txt && \
    if [ "$DEV" = "true" ]; \
        then /py/bin/pip install -r /tmp/requirements.dev.txt; \
    fi && \
    rm -rf /tmp/* && \
    adduser \
        --disabled-password \
        django-user

ENV PATH="/py/bin:$PATH"

USER django-user