rm -fr .venv
poetry config virtualenvs.in-project true
poetry install
# poetry run pip install -U jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry run pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
poetry run python -c "import jax; print(jax.devices())"