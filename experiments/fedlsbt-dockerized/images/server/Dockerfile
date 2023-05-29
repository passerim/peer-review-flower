FROM python:3.10-slim
COPY . $HOME/app
WORKDIR $HOME/app
RUN apt update -y; apt install -y git
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "-u", "server.py"]
