import joblib_modal  # noqa
from joblib import parallel_config, Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


def test_modal_backend():
    with parallel_config(backend="modal", modal_output=True):
        out = Parallel(n_jobs=2)(delayed(lambda x: x * x)(i) for i in range(10))
        assert out == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]


def get_model(use_modal: bool):
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )

    model = RandomizedSearchCV(
        HistGradientBoostingClassifier(),
        param_distributions={
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_leaf_nodes": [10, 20, 30],
            "max_bins": [10, 100, 255],
        },
        n_iter=20,
        cv=5,
        random_state=42,
    )

    if use_modal:
        import sklearn
        import numpy
        import joblib
        import modal
        import scipy

        image = (
            modal.Image.debian_slim()
            .pip_install(f"scikit-learn=={sklearn.__version__}")
            .pip_install(f"numpy=={numpy.__version__}")
            .pip_install(f"joblib=={joblib.__version__}")
            .pip_install(f"scipy=={scipy.__version__}")
        )

        with parallel_config(
            backend="modal",
            n_jobs=-1,
            name="test-joblib",
            image=image,
            modal_output=True,
        ):
            model = model.fit(X, y)
    else:
        model = model.fit(X, y)

    return model


def test_modal_with_image():
    modal_model = get_model(use_modal=True)
    native_model = get_model(use_modal=False)

    assert modal_model.best_params_ == native_model.best_params_
