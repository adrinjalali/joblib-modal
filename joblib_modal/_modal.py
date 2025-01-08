import uuid
from joblib._parallel_backends import (
    FallbackToBackend,
    SequentialBackend,
    ThreadingBackend,
)
from joblib import register_parallel_backend
from joblib._utils import _TracebackCapturingWrapper
import modal


def executor(func, *args, **kwargs):
    return func(*args, **kwargs)


class ModalBackend(ThreadingBackend):
    uses_threads = True
    supports_sharedmem = False

    def __init__(self, *args, name=None, image=None, modal_output=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.image = image
        self.modal_output = modal_output

    def configure(self, n_jobs=1, parallel=None, **_):
        """Build a process or thread pool and return the number of workers"""
        n_jobs = self.effective_n_jobs(n_jobs)

        if n_jobs == 1:
            # Avoid unnecessary overhead and use sequential backend instead.
            raise FallbackToBackend(SequentialBackend(nesting_level=self.nesting_level))

        self.parallel = parallel
        self._n_jobs = n_jobs

        if self.image is None:
            image = modal.Image.debian_slim().pip_install("joblib")
        else:
            image = self.image

        name = self.name or f"modal-joblib-{uuid.uuid4()}"
        self.modal_app = modal.App(name, image=image)

        self.modal_executor = self.modal_app.function()(executor)

        if self.modal_output:
            self.output_ctx = modal.enable_output()
            self.output_ctx.__enter__()
        self.run_ctx = self.modal_app.run()
        self.run_ctx.__enter__()

        return n_jobs

    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs which are going to run in parallel"""
        if n_jobs == 0:
            raise ValueError("n_jobs == 0 in Parallel has no meaning")
        if n_jobs < 0:
            return 1000
        return n_jobs

    def apply_async(self, func, callback=None):
        """Schedule a func to be run"""
        # Here, we need a wrapper to avoid crashes on KeyboardInterruptErrors.
        # We also call the callback on error, to make sure the pool does not
        # wait on crashed jobs.
        return self._get_pool().apply_async(
            _TracebackCapturingWrapper(self.modal_executor.remote),
            (),
            {"func": func},
            callback=callback,
            error_callback=callback,
        )

    def terminate(self):
        if self.modal_output:
            self.output_ctx.__exit__(None, None, None)
        self.run_ctx.__exit__(None, None, None)
        super().terminate()


register_parallel_backend("modal", ModalBackend)
