import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score
from .pipeline import MODEL

def fairness_ok(model, X, y, group, max_gap=0.15):
    pred = model.predict(X)
    tpr = []
    for g in np.unique(group):
        mask = (group == g) & (y == 1)
        tpr.append(pred[mask].mean() if mask.any() else 0.0)
    return (max(tpr) - min(tpr)) <= max_gap

def promote_if_better(run_id, data):
    X, y = data.training_set()
    cand = mlflow.sklearn.load_model(f'runs:/{run_id}/model')
    if not fairness_ok(cand, X, y, data.group):
        return 'BLOCKED: fails fairness gate'
    client = mlflow.MlflowClient()
    try:
        prod = mlflow.sklearn.load_model(f'models:/{MODEL}/Production')
        if accuracy_score(y, cand.predict(X)) <= accuracy_score(y, prod.predict(X)):
            return 'HOLD: not better than incumbent'
    except Exception:
        pass
    mv = mlflow.register_model(f'runs:/{run_id}/model', MODEL)
    client.transition_model_version_stage(MODEL, mv.version, 'Staging')
    return f'PROMOTED v{mv.version} to Staging (passed accuracy + fairness)'
