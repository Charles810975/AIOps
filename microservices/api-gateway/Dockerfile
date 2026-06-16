FROM python:3.11-slim

WORKDIR /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir flask==3.0.0 flask-cors==4.0.0 prometheus-client==0.19.0 requests==2.31.0 gunicorn==21.2.0

COPY app/ /app/

ENV PORT=8080
ENV BACKEND_HOST=frontend
ENV BACKEND_PORT=8080
ENV RATE_LIMIT_REQUESTS=100
ENV RATE_LIMIT_WINDOW=60

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "60", "app:app"]
