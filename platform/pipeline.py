import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from .data import DataLayer, SEED

MODEL = 'PlayerFormClassifier'
mlflow.set_experiment('cricket-mlops-capstone')

def train_and_track(data: DataLayer, max_depth=4) -> str:
    X, y = data.training_set()
    with mlflow.start_run() as run:
        mlflow.log_param('data_version', data.data_version)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('seed', SEED)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=SEED).fit(X, y)
        mlflow.log_metric('accuracy', accuracy_score(y, model.predict(X)))
        mlflow.sklearn.log_model(model, 'model')
        return run.info.run_id
