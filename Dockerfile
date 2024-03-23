FROM python:3.12.1
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD ["uvicorn", "task2:app", "--host", "0.0.0.0", "--port", "$PORT", "--reload"]
