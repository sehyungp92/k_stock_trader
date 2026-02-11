FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies from root
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir pykrx>=1.0.0

# Copy shared libs
COPY kis_core/ /app/kis_core/
COPY oms_client/ /app/oms_client/

# Copy OMS domain types (Intent, etc.) - needed by oms_client
COPY oms/__init__minimal.py /app/oms/__init__.py
COPY oms/intent.py /app/oms/intent.py

# Copy strategy code
COPY strategy_nulrimok/ /app/strategy_nulrimok/

# Copy backfill script
COPY scripts/backfill_lrs.py /app/scripts/backfill_lrs.py

ENV PYTHONPATH=/app

CMD ["python", "-m", "strategy_nulrimok.main"]
