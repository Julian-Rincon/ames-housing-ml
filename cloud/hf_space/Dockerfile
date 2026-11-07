# SAVI agent API + web UI on a free Hugging Face Docker Space (CPU basic, $0).
# Pure standard library: no pip install needed. Data artifacts live in ./data.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user . /app
USER user
ENV SAVI_DATA_DIR=/app/data PYTHONUNBUFFERED=1
STOPSIGNAL SIGINT
EXPOSE 7860
CMD ["python", "local_server.py", "--host", "0.0.0.0", "--port", "7860"]
