from platform.data import DataLayer
from platform.registry import fairness_ok

def test_data_is_reproducible():
    a, b = DataLayer(), DataLayer()
    assert (a.X == b.X).all()
    assert (a.y == b.y).all()

def test_fairness_gate_returns_bool():
    d = DataLayer()
    from sklearn.tree import DecisionTreeClassifier
    m = DecisionTreeClassifier(max_depth=4, random_state=1983).fit(d.X, d.y)
    assert isinstance(fairness_ok(m, d.X, d.y, d.group), bool)
