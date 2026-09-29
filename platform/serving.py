import mlflow, mlflow.sklearn
import numpy as np
import shap
from .pipeline import MODEL
from .data import FEATURES
from .governance import Governance

class ServingLayer:
    def __init__(self, reference_features):
        self.model = mlflow.sklearn.load_model(f'models:/{MODEL}/Production')
        self.explainer = shap.TreeExplainer(self.model)
        self.governance = Governance(reference_features)

    def predict(self, client_id, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (4,): return {'error': 'invalid_input'}
        block = self.governance.guard(client_id, x)
        if block: return {'error': block}
        pred = int(self.model.predict([x])[0])
        values = self.explainer.shap_values([x])
        sv = np.asarray(values)
        if sv.ndim == 3: sv = sv[0, :, -1]
        elif sv.ndim == 2: sv = sv[0]
        reasons = sorted(zip(FEATURES, sv), key=lambda t: -abs(float(t[1])))[:3]
        result = {'in_form': bool(pred), 'top_reasons': [{'f': f, 'c': round(float(c), 3)} for f, c in reasons]}
        self.governance.audit(x, pred)
        return result
