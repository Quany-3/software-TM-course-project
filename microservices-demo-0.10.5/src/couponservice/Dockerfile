FROM python:3.14.2-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /couponservice

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT="50053"
EXPOSE 50053

ENTRYPOINT ["python", "coupon_server.py"]
