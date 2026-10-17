# On-premise image for the grading platform (CPU inference).
#
#   docker compose up --build          see docker-compose.yml
#
# The image holds code only. Weights, evaluation evidence, the audit database
# and encrypted images live on mounted volumes, and the image-encryption key
# comes from the environment (DR_DATA_KEY) -- never baked into a layer.
#
# TLS is terminated by the hospital's reverse proxy in front of this
# container; the service refuses a non-local bind unless told so explicitly
# (--behind-tls-proxy) and also requires the not-a-medical-device
# acknowledgement. Read INTENDED_USE.md section 7 before deploying.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code changes do not invalidate this layer.
COPY requirements-cpu.txt requirements.txt requirements-serve.txt ./
RUN pip install -r requirements-cpu.txt \
 && pip install -r requirements.txt -r requirements-serve.txt

COPY *.py ./
COPY webapp/ webapp/
COPY INTENDED_USE.md MODEL_CARD.md ./

# Run unprivileged. The mounted volumes must be writable by this user.
RUN useradd --create-home --uid 10001 dr \
 && mkdir -p artifacts checkpoints secrets \
 && chown -R dr:dr /app
USER dr

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["python", "service.py", "--host", "0.0.0.0", "--port", "8000", \
     "--behind-tls-proxy", "--i-understand-this-is-not-a-medical-device"]
