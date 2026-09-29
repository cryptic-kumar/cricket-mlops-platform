import mlflow
from .data import DataLayer
from .pipeline import train_and_track, MODEL
from .registry import promote_if_better
from .monitoring import MonitoringLayer
from .control_plane import ControlPlane

def main():
    data = DataLayer()
    run_id = train_and_track(data)
    promo = promote_if_better(run_id, data)
    print(promo)
    print('Data version:', data.data_version)
    print('Run:', run_id)
    print('Monitoring:', MonitoringLayer(data.reference_features()).drift_detected())
    print('Control plane:', ControlPlane(data, MonitoringLayer(data.reference_features())).maybe_retrain())
    print('Next: approve Staging -> Production in MLflow before starting the serving API.')

if __name__ == '__main__': main()
