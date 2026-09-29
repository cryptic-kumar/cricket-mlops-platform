import json, os
import numpy as np
from scipy import stats

class MonitoringLayer:
    def __init__(self, reference_features, audit_path='audit/decisions.log'):
        self.ref = np.asarray(reference_features)
        self.audit_path = audit_path

    def drift_detected(self, min_n=100):
        if not os.path.exists(self.audit_path): return False
        with open(self.audit_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        if len(rows) < min_n: return False
        live = np.array([r['x'][0] for r in rows[-500:]])
        return bool(stats.ks_2samp(self.ref, live).pvalue < 0.05)
