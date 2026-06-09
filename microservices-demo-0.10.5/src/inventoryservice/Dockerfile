FROM python:3.14.2-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /inventoryservice

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT="50052"
EXPOSE 50052

ENTRYPOINT ["python", "inventory_server.py"]
