rm -fr .venv
poetry install
poetry run pip install -U jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry run python -c "import jax; print(jax.devices())"