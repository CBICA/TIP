FROM --platform=linux/amd64 python:3.9-slim-buster

LABEL IMAGE="Result-Generator"
LABEL VERSION="0.0.1"
LABEL CI_IGNORE="True"

RUN mkdir /src /logs /refs
COPY files /src
COPY logos /logos
COPY refs /refs
COPY requirements.txt /src 

RUN apt-get update && \
    apt-get -y install libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info libgl1 git libc6 && \
    cd src && \
    pip install -r requirements.txt && \
    git clone https://github.com/CBICA/spare_score && \
    cd spare_score && \
    pip install .

CMD ["python3","-u","/src/start.py"]


