FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --upgrade pip

RUN pip install tensorflow==2.7.0
RUN pip install tensorflow-quantum==0.7.2

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "file.sh"]
