FROM python:3.11

RUN pip install gradio>=3.36.1
RUN pip install google-cloud-aiplatform==1.50.0 google-cloud-logging

COPY ./app /app

WORKDIR /app

CMD ["python", "app.py"]
