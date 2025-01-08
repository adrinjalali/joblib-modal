from joblib_modal import enable_joblib_modal  # noqa
from sklearn.model_selection import GridSearchCV
from joblib import parallel_config

...
est = GridSearchCV(...)
with parallel_config(backend="modal", inner_max_num_threads=2):
    est.fit(X, y)
