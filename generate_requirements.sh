python -m piptools compile pyproject.toml --extra dev --output-file requirements-dev.txt
python -m piptools compile pyproject.toml --extra cuda128 --output-file requirements-cuda128.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -m piptools compile pyproject.toml --extra cpu --output-file requirements-cpu.txt