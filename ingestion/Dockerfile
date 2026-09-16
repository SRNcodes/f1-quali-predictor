FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jolpica_client.py fetch_data.py ./

ENTRYPOINT ["python", "fetch_data.py"]
