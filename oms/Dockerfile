FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies from root
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy shared libraries
COPY kis_core/ /app/kis_core/
COPY strategy_common/ /app/strategy_common/
COPY instrumentation/__init__.py /app/instrumentation/__init__.py
COPY instrumentation/src/ /app/instrumentation/src/

# Copy OMS
COPY oms/ /app/oms/

ENV PYTHONPATH=/app

CMD ["python", "-m", "oms.server"]
