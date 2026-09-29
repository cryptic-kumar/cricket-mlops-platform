import json, os, time
from collections import defaultdict, deque
import numpy as np

class Governance:
    def __init__(self, reference_features, audit_path='audit/decisions.log'):
        self.ref = np.asarray(reference_features)
        self.history = defaultdict(lambda: deque(maxlen=100))
        self.audit_path = audit_path

    def guard(self, client_id, x):
        now = time.time(); h = self.history[client_id]
        while h and now - h[0] > 1.0: h.popleft()
        h.append(now)
        if len(h) > 5: return 'rate_limited'
        lo, hi = self.ref.min(), self.ref.max()
        if x[0] < lo - 0.2*(hi-lo) or x[0] > hi + 0.2*(hi-lo):
            return 'out_of_distribution'
        return None

    def audit(self, x, pred):
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        with open(self.audit_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'ts': time.time(), 'x': list(x), 'pred': int(pred)}) + '\n')
