FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install tesseract-ocr -y \
    && apt-get install tesseract-ocr-eng -y \
    && apt-get install tesseract-ocr-swe

RUN apt-get install -y openjdk-8-jre openjdk-8-jdk
RUN apt-get install poppler-utils -y

RUN apt-get update -y
RUN apt-get install -y git
RUN apt-get install -y apt-utils
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update -y

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv

WORKDIR /InvoiceDataExtraction_new

COPY . /InvoiceDataExtraction_new

RUN python3.6 -m compileall -b /InvoiceDataExtraction_new
RUN find . -type f -name '*.py' -delete
RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirement.txt
RUN python3.6 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


ENTRYPOINT [ "python3.6" ]

CMD [ "test3.pyc" ]

