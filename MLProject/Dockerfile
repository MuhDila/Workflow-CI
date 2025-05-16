FROM python:3.12

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir mlflow scikit-learn joblib numpy

CMD ["python", "modelling.py"]
