# Usage

This library allows you to use `modal <https://modal.com/>`_ as a joblib backend.

It is particularly useful if you're running a ``GridSearchCV`` or similar with
`scikit-learn <https://scikit-learn.org/>`_.

For direct joblib usage you can do:

.. code-block:: python
  :linenos:
  
    # This is needed to register the modal backend
    import joblib_modal  # noqa
    from joblib import parallel_config, Parallel, delayed

    with parallel_config(backend="modal", name="my-test-job",modal_output=True):
        out = Parallel(n_jobs=2)(delayed(lambda x: x * x)(i) for i in range(10))
        assert out == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

And for a scikit-learn usage, you can do like the following:

.. code-block:: python
  :linenos:
  
    import joblib_modal  # noqa
    import modal
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import HistGradientBoostingClassifier

    image = (
        modal.Image.debian_slim()
        .pip_install(f"scikit-learn=={sklearn.__version__}")
        .pip_install(f"numpy=={numpy.__version__}")
        .pip_install(f"joblib=={joblib.__version__}")
        .pip_install(f"scipy=={scipy.__version__}")
    )

    param_grid = {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
    clf = HistGradientBoostingClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=2)
    with parallel_config(
        backend="modal",
        n_jobs=-1,
        name="test-joblib",
        image=image,
        modal_output=True,
    ):
        grid_search.fit(X, y)
