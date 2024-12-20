FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libgomp1 libglib2.0-0 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1000 user && \
    chown -R user:user /app

COPY --chown=user . .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

USER user

RUN paddleocr --image_dir ./image.png --use_angle_cls true --use_gpu false  --lang en

EXPOSE 8000

CMD ["python", "app.py"]