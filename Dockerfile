FROM ubuntu:20.04

ENV DEBIAN_FRONTEND "noninteractive"
ENV LANG "ja_JP.UTF-8"
ENV PYTHONIOENCODING "utf-8"

RUN apt update -y \
      && apt install -y \
            language-pack-ja \
            build-essential \
            git \
            wget \
            libmecab-dev \
            python3 \
            python3-dev \
            python3-pip \
      && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip
ARG project_dir=/work/
WORKDIR $project_dir
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install fastapi && pip install uvicorn
RUN python3 -m spacy download ja_core_news_md
COPY . $project_dir

CMD ["uvicorn", "jel.api.server:app", "--reload", "--port", "8000", "--host", "0.0.0.0", "--log-level", "trace"]