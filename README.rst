Installation
============

.. code-block:: bash

    pip install joblib-modal

Usage
=====

This library allows you to use `modal <https://modal.com/>`_ as a joblib backend.

It is particularly useful if you're running a ``GridSearchCV`` or similar with
`scikit-learn <https://scikit-learn.org/>`_.

For direct joblib usage you can do:

.. code-block:: python
  
    # This is needed to register the modal backend
    import joblib_modal  # noqa
    from joblib import parallel_config, Parallel, delayed

    with parallel_config(backend="modal", name="my-test-job",modal_output=True):
        out = Parallel(n_jobs=2)(delayed(lambda x: x * x)(i) for i in range(10))
        assert out == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

And for a scikit-learn usage, you can do like the following:

.. code-block:: python
  
    import joblib_modal  # noqa
    import modal
    import numpy 
    import joblib 
    import scipy
    import sklearn
    from joblib import parallel_config
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.datasets import make_classification

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
    X, y = make_classification()

    with parallel_config(
        backend="modal",
        n_jobs=-1,
        name="test-joblib",
        image=image,
        modal_output=True,
    ):
        grid_search.fit(X, y)

API
===

The backend is used via the ``joblib.parallel_config`` context manager, and in the
case of this backend, the signature is:

.. code-block:: python

    with parallel_config(
        backend="modal",
        n_jobs: int = 1,
        name: str = None,
        modal_output: bool = False,
        image: modal.Image = None,
        modal_function_kwargs: dict = None,
    ):
        ...

- ``n_jobs``: The number of jobs to run in parallel. This specifies the maximum number
  of concurrent jobs submitted to `modal`_. Note that you're limited by your maximum
  number of concurrent jobs in your modal account, and if that is exceeded, the jobs
  will be queued up and run in order.
- ``name``: The name of the modal app. If not provided,
  ``f"modal-joblib-{uuid.uuid4()}"`` is used.
- ``modal_output``: Whether to enable modal output. If enabled, the output of the jobs
  will be captured and returned. This is equivalent to using the
  ``modal.enable_output()`` context manager.
- ``image``: The modal image to use for the jobs. If not provided, a debian slim image
  with ``joblib`` installed is used. Your image should always have ``joblib`` installed
  and you should ideally replicate your local environment as closely as possible.
  See `modal.Image <https://modal.com/docs/reference/modal.Image>`_ for more details.
- ``modal_function_kwargs``: The kwargs to pass to the modal ``app.function()``
  decorator. See `modal.App.function() <https://modal.com/docs/reference/modal.App>`_
  for more details.
