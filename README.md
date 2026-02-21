# SHELL — SHELL Highlights Epithelium and Lumen Locations

[![Build](https://github.com/laviolette-lab/shell/actions/workflows/build.yml/badge.svg)](https://github.com/laviolette-lab/shell/actions/workflows/build.yml)
[![Tests](https://github.com/laviolette-lab/shell/actions/workflows/pytest.yml/badge.svg)](https://github.com/laviolette-lab/shell/actions/workflows/pytest.yml)
[![Lint](https://github.com/laviolette-lab/shell/actions/workflows/lint.yml/badge.svg)](https://github.com/laviolette-lab/shell/actions/workflows/lint.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/shell.svg)](https://pypi.org/project/shell)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/shell.svg)](https://pypi.org/project/shell)

-----

A whole-slide image segmentation pipeline for H&E-stained histopathology.
SHELL uses Macenko colour deconvolution and a SegResNetVAE to segment
epithelium and lumen/stroma regions.

Model weights are **bundled with the package** — no separate download required.

## Label Map

| Value | Label |
|-------|-------|
| 0 | Background / White |
| 1 | Epithelium |
| 2 | Stroma |

## Quick Start

### Installation

```console
pip install shell
```

> **Python 3.10 – 3.12** is required (constrained by the OMERO dependency).

To use OMERO support (fetching images from an OMERO server):

```console
pip install shell[omero]
```

### CLI — Local WSI Inference

The bundled model is used automatically — no `--model-path` needed:

```console
shell infer --input slide.tiff --output prediction.tiff
```

To use a specific bundled model version:

```console
shell infer --input slide.tiff --output prediction.tiff --model-version v1
```

To use your own weights instead:

```console
shell infer --input slide.tiff --output prediction.tiff --model-path /path/to/custom.pth
```

### CLI — OMERO Inference

```console
shell infer-omero \
    --host $OMERO_SERVER --port $OMERO_PORT \
    --username $OMERO_USERNAME --password $OMERO_PASSWORD \
    --image-id 12345 \
    --output pred_12345.tiff
```

### As a Library

```python
from shell.infer_wsi import infer_wsi

# Uses the bundled latest model automatically
label_map = infer_wsi(
    input_path="slide.tiff",
    output_path="prediction.tiff",
)
```

You can also load the model directly:

```python
from shell.model import build_model

# Bundled latest
model = build_model(device="cuda")

# Specific bundled version
model = build_model(model_version="v1", device="cuda")

# Custom weights file
model = build_model("/path/to/custom.pth", device="cuda")
```

## Model Versioning

Model weights live in `src/shell/weights/` and are registered in
`shell.model.MODEL_REGISTRY`. The `LATEST_MODEL` constant controls which
version is loaded by default.

| Version | File | Notes |
|---------|------|-------|
| `v1` | `model_v1.pth` | Initial release — SegResNetVAE trained on H&E |

### Adding a New Model

1. Place the new `.pth` file in `src/shell/weights/`.
2. Add an entry to `MODEL_REGISTRY` in `src/shell/model.py`:
   ```python
   MODEL_REGISTRY: dict[str, str] = {
       "v1": "model_v1.pth",
       "v2": "model_v2.pth",  # new
   }
   ```
3. Update `LATEST_MODEL`:
   ```python
   LATEST_MODEL: str = "v2"
   ```
4. Bump the package version and release.

## Development Setup

**Prerequisites:** Python 3.10–3.12 and [Hatch](https://hatch.pypa.io/latest/install/).

```console
git clone https://github.com/laviolette-lab/shell.git
cd shell
pip install hatch
```

Optionally install pre-commit hooks:

```console
pip install pre-commit
pre-commit install
```

### Common Commands

| Task | Command |
|------|---------|
| Run tests | `hatch run test:test` |
| Tests + coverage | `hatch run test:cov` |
| Lint | `hatch run lint:check` |
| Format | `hatch run lint:format` |
| Auto-fix lint | `hatch run lint:fix` |
| Format + fix + lint | `hatch run lint:all` |
| Type check | `hatch run types:check` |
| Build docs | `hatch run docs:build-docs` |
| Serve docs | `hatch run docs:serve-docs` |
| Build wheel | `hatch build` |
| Clean artifacts | `make clean` |

## Publishing

Releases are fully automated. Creating a GitHub Release triggers the
`publish.yml` workflow, which:

1. Builds a wheel and sdist and publishes them to **PyPI** via trusted
   publishing (OIDC — no API tokens needed).
2. Builds **standalone binaries** for Linux (x86_64) and macOS (x86_64 +
   arm64) using [Nuitka](https://nuitka.net/) with Python 3.12.
3. Uploads the binaries as **release assets** on the GitHub Release.

### How to Release

```bash
# 1. Bump the version in src/shell/__about__.py
# 2. Commit and tag
git add -A
git commit -m "release: v0.2.0"
git tag v0.2.0
git push && git push --tags

# 3. Create a GitHub Release from the tag
gh release create v0.2.0 --generate-notes
```

### One-Time Setup (PyPI)

1. Go to <https://pypi.org/manage/account/publishing/>.
2. Add a pending publisher:
   - **PyPI project name:** `shell`
   - **Owner:** `laviolette-lab`
   - **Repository:** `shell`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`

### One-Time Setup (GitHub)

1. In repository **Settings → Environments**, create an environment named
   **`pypi`** (optionally with manual approval protection).

## Project Structure

```text
shell/
├── src/
│   └── shell/                   # Package source
│       ├── __init__.py          # Public API & version export
│       ├── __about__.py         # Version string
│       ├── cli.py               # CLI entry point
│       ├── model.py             # SegResNetVAE model helpers & registry
│       ├── preprocessing.py     # Macenko deconvolution & EHO transform
│       ├── inference.py         # Sliding-window inference
│       ├── infer_wsi.py         # Local WSI pipeline
│       ├── infer_omero_wsi.py   # OMERO WSI pipeline (optional dep)
│       ├── weights/             # Bundled model weight files
│       │   └── model_v1.pth
│       └── py.typed             # PEP 561 marker
├── tests/                       # pytest test suite
├── docs/                        # MkDocs source files
├── .github/
│   └── workflows/
│       ├── build.yml            # CI: build wheel on push/PR
│       ├── pytest.yml           # CI: run tests
│       ├── lint.yml             # CI: ruff lint + format check
│       └── publish.yml          # CD: PyPI publish + Nuitka binaries
├── pyproject.toml               # All project & tool configuration
├── Dockerfile                   # Multi-stage build (hatch / dev / prod)
├── Makefile                     # Dev shortcuts
└── README.md
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## License

`shell` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.