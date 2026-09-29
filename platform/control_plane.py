import logging
from .pipeline import train_and_track
from .registry import promote_if_better

log = logging.getLogger('platform')

class ControlPlane:
    def __init__(self, data, monitoring):
        self.data, self.monitoring = data, monitoring

    def train_register_deploy(self, max_depth=4):
        run_id = train_and_track(self.data, max_depth)
        result = promote_if_better(run_id, self.data)
        if 'PROMOTED' not in result:
            return {'status': result}
        return {'status': result, 'note': 'awaiting approval for Production',
                'wired': ['lineage','monitoring','fairness_gate','explanations','api_defence']}

    def maybe_retrain(self):
        if self.monitoring.drift_detected():
            log.info('Drift detected -> retraining + re-gating')
            return self.train_register_deploy()
        return {'status': 'no retrain needed'}
