FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY deployment/ /app/deployment/
COPY strategy_common/ /app/strategy_common/
COPY strategy_kalcb/ /app/strategy_kalcb/
COPY strategy_olr/ /app/strategy_olr/
COPY kis_core/ /app/kis_core/
RUN printf '%s\n' '"""Minimal kis_core package for OLR/KALCB runtime containers."""' > /app/kis_core/__init__.py
COPY oms_client/ /app/oms_client/
COPY scripts/check_live_readiness_manifest.py /app/scripts/check_live_readiness_manifest.py
COPY scripts/generate_olr_kalcb_artifacts.py /app/scripts/generate_olr_kalcb_artifacts.py
COPY scripts/replay_paper_session.py /app/scripts/replay_paper_session.py
COPY scripts/run_olr_kalcb_runtime_session.py /app/scripts/run_olr_kalcb_runtime_session.py
COPY scripts/summarize_paper_parity.py /app/scripts/summarize_paper_parity.py

RUN mkdir -p /app/oms
COPY oms/__init__minimal.py /app/oms/__init__.py
COPY oms/intent.py /app/oms/intent.py
RUN chmod +x /app/deployment/olr_kalcb/runtime_service_entrypoint.sh

ENV PYTHONPATH=/app

CMD ["sh", "/app/deployment/olr_kalcb/runtime_service_entrypoint.sh"]
