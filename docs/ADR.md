# `docs`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`docs` owns the Sphinx documentation source tree and the build entry points for published PyTorch reference documentation. It defines how doc pages, generated API pages, notebooks, and static assets are assembled into the site at `docs.pytorch.org`.

## Key Files

| File | Purpose |
|---|---|
| `README.md` | Points contributors at the documentation workflow in `CONTRIBUTING.md` |
| `Makefile` | Defines the standard `sphinx-build` targets, asset-generation helpers, `html-stable`, and a local HTTP server |
| `source/conf.py` | Configures Sphinx extensions, theme options, autosummary generation, notebook execution policy, and site metadata |
| `libtorch.rst` | Top-level documentation source for the libtorch C++ distribution |
| `requirements.txt` | Lists Python packages needed by the documentation toolchain |

## Public Interface

The component exposes `make help`, `make html`, `make html-stable`, `make serve`, and the catch-all `%` target in `Makefile`. `source/conf.py` exports configuration symbols such as `extensions`, `myst_enable_extensions`, `autosummary_generate`, `html_theme_options`, and `html_context` that Sphinx loads at build time.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | `source/conf.py` imports `torch`, reads `torch.__version__`, and builds API documentation against the installed Python package |
| [torch/onnx](torch/onnx/ADR.md) | depends-on | `Makefile` exposes the `opset` and `exportdb` helper targets that generate ONNX- and export-related documentation content |

## Runtime Behaviour

Running `make html` or any other unknown target enters the `%: Makefile figures onnx opset exportdb` rule, which first generates derived assets and then dispatches to `sphinx-build -M`. `Makefile` also defines `html-stable` by exporting `RELEASE=1`, and `serve` changes into `build/html` and starts `python -m http.server` for local preview.

At import time, `source/conf.py` imports `torch`, attempts to import `torchvision`, enables extensions like `sphinx.ext.autodoc`, `myst_nb`, `sphinx_copybutton`, and `sphinx_sitemap`, and sets `nb_execution_mode = "off"` so the docs build does not re-execute notebooks. The same file sets `autosummary_generate = True`, configures the `pytorch_sphinx_theme2` theme, and fills `html_context` with repository metadata for edit links and version switching.

## Performance Profile

- **Allocation sites** - Sphinx allocates large doctrees and generated autosummary pages, and the helper targets create derived images and generated `.rst` content before the final build.
- **Synchronization costs** - The build uses `SPHINXOPTS ?= -j $(shell nproc) -WT --keep-going`, so page rendering can parallelize across CPUs, but prerequisite asset-generation targets run before Sphinx starts.
- **Data movement** - The build writes HTML into `build/html`, doctrees into `build/doctrees`, generated pages into `source/generated`, and local preview serves those files directly from disk.
- **Redundant or repeated work** - `nb_execution_mode = "off"` removes repeated notebook execution from the docs build, and `clean` explicitly deletes `build/html`, `build/doctrees`, `source/generated`, and `build/auto_gen_aten_op_list.csv` to force a fresh rebuild when needed.

## Design Rationale

PyTorch keeps documentation as a first-class source component so API reference, tutorials, and generated operator pages evolve with the code. The Makefile plus `conf.py` split lets contributors use familiar Sphinx commands while still baking PyTorch-specific behavior like version switching, generated assets, and notebook policy into one shared configuration.
