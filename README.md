# Cricket MLOps Platform

A compact implementation of the course capstone: versioned cricket data, tracked training, MLflow registry, gated promotion, defended/explained serving, drift monitoring, audit logging, and a self-service control plane.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
pip install -r requirements.txt
python -m platform.demo
```

The demo uses deterministic synthetic cricket features from the capstone brief, so it can run without downloading a real dataset.

## Components

- `platform/data.py` - deterministic versioned data + reference features
- `platform/pipeline.py` - training + MLflow tracking
- `platform/registry.py` - accuracy/fairness promotion gate
- `platform/serving.py` - validation, rate limiting, SHAP explanations, audit
- `platform/monitoring.py` - KS drift detection
- `platform/governance.py` - API defence + audit helpers
- `platform/control_plane.py` - self-service lifecycle + retraining
- `platform/demo.py` - end-to-end demonstration
- `infra/` - Terraform starter for the infrastructure-as-code requirement
- `.github/workflows/ci.yml` - CI gate

## Production path

Use Kubernetes, object storage, a managed database, MLflow server, Prometheus/Grafana, and a real feature/data versioning system behind the same interfaces. The local implementation intentionally keeps the moving parts small enough to understand and demonstrate.
