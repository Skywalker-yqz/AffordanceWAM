# Python dependency license inventory — AffordanceWAM

This inventory records upstream-reported license metadata for the software explicitly declared by this repository. The repository license does not replace the licenses of these packages. See [Third-party notices](../THIRD_PARTY_NOTICES.md) for source-code attributions, external repositories, model checkpoints, benchmark assets, and non-Python software.

**Metadata checked:** 2026-10-18. **Scope:** 62 normalized PyPI package names and 66 declared package/version entries, plus any direct Git/URL requirements listed below.

This is a declaration inventory, not a complete resolved or installed SBOM. Optional environment snapshots can contain different versions and unused development tools; their inclusion does not mean that every package is required by the core model. This inventory does not install packages, certify version availability or compatibility, or establish that an optional component is redistributed here.

The table uses each exact release's official [PyPI JSON metadata](https://docs.pypi.org/api/json/). An SPDX expression is reproduced when present; otherwise a short license field or classifier is reported. Classifiers can be less precise than the actual license text. Where multiple labels are reported without an SPDX expression, no AND/OR relationship is inferred. Empty metadata and unavailable releases are explicitly marked; a different release's license is not silently substituted.

The linked distribution and upstream license files control. Wheels and source archives can include additional licenses for bundled native libraries, fonts, codecs, and other code. In particular, NVIDIA CUDA packages have NVIDIA terms, and FFmpeg-related packages can bundle components with additional terms. Preserve the actual distribution's license and notice files when redistributing it.

## Declaration sources

- [env/install.sh](../env/install.sh)
- [env/requirements-dev.txt](../env/requirements-dev.txt)
- [env/requirements.txt](../env/requirements.txt)
- [pyproject.toml](../pyproject.toml)

Some declarations use ranges or an unversioned install. The table below covers pinned versions present elsewhere in the repository; it does not extend those license checks to every version permitted by a range. Check the distribution actually selected by your installer.

| Declaration | Source |
| --- | --- |
| `setuptools>=68` | [pyproject.toml](../pyproject.toml) |
| `wheel` | [pyproject.toml](../pyproject.toml) |
| `av>=13.1,<14` | [pyproject.toml](../pyproject.toml) |
| `einops>=0.7,<0.9` | [pyproject.toml](../pyproject.toml) |
| `ftfy>=6.3,<7` | [pyproject.toml](../pyproject.toml) |
| `huggingface-hub>=0.29,<0.30` | [pyproject.toml](../pyproject.toml) |
| `imageio>=2.37,<3` | [pyproject.toml](../pyproject.toml) |
| `imageio-ffmpeg>=0.5,<0.7` | [pyproject.toml](../pyproject.toml) |
| `matplotlib>=3.9,<3.11` | [pyproject.toml](../pyproject.toml) |
| `numpy>=1.26,<2` | [pyproject.toml](../pyproject.toml) |
| `omegaconf>=2.3,<2.4` | [pyproject.toml](../pyproject.toml) |
| `opencv-python-headless>=4.10,<4.11` | [pyproject.toml](../pyproject.toml) |
| `pandas>=2.2,<2.3` | [pyproject.toml](../pyproject.toml) |
| `pillow>=10.3,<12` | [pyproject.toml](../pyproject.toml) |
| `protobuf>=5.29,<6` | [pyproject.toml](../pyproject.toml) |
| `regex>=2024.11.6` | [pyproject.toml](../pyproject.toml) |
| `safetensors>=0.5,<0.6` | [pyproject.toml](../pyproject.toml) |
| `sentencepiece>=0.2,<0.3` | [pyproject.toml](../pyproject.toml) |
| `tqdm>=4.67,<5` | [pyproject.toml](../pyproject.toml) |
| `transformers>=4.49,<4.50` | [pyproject.toml](../pyproject.toml) |
| `typing-extensions>=4.12,<5` | [pyproject.toml](../pyproject.toml) |

## Declared packages

Each version in the source column links to that release's metadata. Rows are grouped only when the reported license summary is identical.

| Package | Declared version(s) | Reported license | Metadata source |
| --- | --- | --- | --- |
| `annotated-types` | `0.8.0` | MIT | [0.8.0](https://pypi.org/pypi/annotated-types/0.8.0/json) |
| `antlr4-python3-runtime` | `4.9.3` | BSD | [4.9.3](https://pypi.org/pypi/antlr4-python3-runtime/4.9.3/json) |
| `av` | `13.1.0` | BSD | [13.1.0](https://pypi.org/pypi/av/13.1.0/json) |
| `build` | `1.2.2.post1` | MIT License | [1.2.2.post1](https://pypi.org/pypi/build/1.2.2.post1/json) |
| `certifi` | `2026.7.22` | MPL-2.0 | [2026.7.22](https://pypi.org/pypi/certifi/2026.7.22/json) |
| `charset-normalizer` | `3.5.2` | MIT | [3.5.2](https://pypi.org/pypi/charset-normalizer/3.5.2/json) |
| `contourpy` | `1.3.2` | BSD License (classifier; full notices in metadata) | [1.3.2](https://pypi.org/pypi/contourpy/1.3.2/json) |
| `cycler` | `0.12.1` | BSD License (classifier; full notices in metadata) | [0.12.1](https://pypi.org/pypi/cycler/0.12.1/json) |
| `deepspeed` | `0.16.9` | Apache Software License 2.0 | [0.16.9](https://pypi.org/pypi/deepspeed/0.16.9/json) |
| `einops` | `0.8.1` | MIT | [0.8.1](https://pypi.org/pypi/einops/0.8.1/json) |
| `filelock` | `3.32.3` | MIT | [3.32.3](https://pypi.org/pypi/filelock/3.32.3/json) |
| `fonttools` | `4.65.0` | MIT | [4.65.0](https://pypi.org/pypi/fonttools/4.65.0/json) |
| `fsspec` | `2026.7.0` | Unspecified in release metadata; consult upstream | [2026.7.0](https://pypi.org/pypi/fsspec/2026.7.0/json), [upstream](https://github.com/fsspec/filesystem_spec) |
| `ftfy` | `6.3.1` | Apache-2.0 | [6.3.1](https://pypi.org/pypi/ftfy/6.3.1/json) |
| `hjson` | `3.1.0` | MIT License | [3.1.0](https://pypi.org/pypi/hjson/3.1.0/json) |
| `huggingface-hub` | `0.29.3` | Apache | [0.29.3](https://pypi.org/pypi/huggingface-hub/0.29.3/json) |
| `idna` | `3.20` | BSD-3-Clause | [3.20](https://pypi.org/pypi/idna/3.20/json) |
| `imageio` | `2.37.0` | BSD-2-Clause | [2.37.0](https://pypi.org/pypi/imageio/2.37.0/json) |
| `imageio-ffmpeg` | `0.6.0` | BSD-2-Clause | [0.6.0](https://pypi.org/pypi/imageio-ffmpeg/0.6.0/json) |
| `jinja2` | `3.1.6` | BSD License | [3.1.6](https://pypi.org/pypi/jinja2/3.1.6/json) |
| `kiwisolver` | `1.5.1` | BSD License (classifier; full notices in metadata) | [1.5.1](https://pypi.org/pypi/kiwisolver/1.5.1/json) |
| `markupsafe` | `3.0.3` | BSD-3-Clause | [3.0.3](https://pypi.org/pypi/markupsafe/3.0.3/json) |
| `matplotlib` | `3.10.1` | Python Software Foundation License (classifier; full notices in metadata) | [3.10.1](https://pypi.org/pypi/matplotlib/3.10.1/json) |
| `mpmath` | `1.3.0` | BSD | [1.3.0](https://pypi.org/pypi/mpmath/1.3.0/json) |
| `msgpack` | `1.1.0` | Apache 2.0 | [1.1.0](https://pypi.org/pypi/msgpack/1.1.0/json) |
| `networkx` | `3.4.2` | BSD License | [3.4.2](https://pypi.org/pypi/networkx/3.4.2/json) |
| `ninja` | `1.11.1.4` | Apache Software License; BSD License | [1.11.1.4](https://pypi.org/pypi/ninja/1.11.1.4/json) |
| `numpy` | `1.26.4` | BSD License (classifier; full notices in metadata) | [1.26.4](https://pypi.org/pypi/numpy/1.26.4/json) |
| `omegaconf` | `2.3.0` | BSD License | [2.3.0](https://pypi.org/pypi/omegaconf/2.3.0/json) |
| `opencv-python-headless` | `4.10.0.84` | Apache 2.0 | [4.10.0.84](https://pypi.org/pypi/opencv-python-headless/4.10.0.84/json) |
| `packaging` | `24.2` | Apache Software License; BSD License | [24.2](https://pypi.org/pypi/packaging/24.2/json) |
| `pandas` | `2.2.3` | BSD License (classifier; full notices in metadata) | [2.2.3](https://pypi.org/pypi/pandas/2.2.3/json) |
| `pillow` | `11.1.0` | MIT-CMU | [11.1.0](https://pypi.org/pypi/pillow/11.1.0/json) |
| `pip` | `25.0.1` | MIT | [25.0.1](https://pypi.org/pypi/pip/25.0.1/json) |
| `protobuf` | `5.29.3` | 3-Clause BSD License | [5.29.3](https://pypi.org/pypi/protobuf/5.29.3/json) |
| `psutil` | `7.0.0` | BSD-3-Clause | [7.0.0](https://pypi.org/pypi/psutil/7.0.0/json) |
| `py-cpuinfo` | `9.0.0` | MIT | [9.0.0](https://pypi.org/pypi/py-cpuinfo/9.0.0/json) |
| `pydantic` | `2.10.6` | MIT License | [2.10.6](https://pypi.org/pypi/pydantic/2.10.6/json) |
| `pydantic-core` | `2.27.2` | MIT | [2.27.2](https://pypi.org/pypi/pydantic-core/2.27.2/json) |
| `pyparsing` | `3.3.3` | MIT | [3.3.3](https://pypi.org/pypi/pyparsing/3.3.3/json) |
| `pytest` | `8.3.5` | MIT | [8.3.5](https://pypi.org/pypi/pytest/8.3.5/json) |
| `pytest-xdist` | `3.6.1` | MIT License (classifier; full notices in metadata) | [3.6.1](https://pypi.org/pypi/pytest-xdist/3.6.1/json) |
| `python-dateutil` | `2.9.0.post0` | Dual License (classifiers: Apache Software License; BSD License) | [2.9.0.post0](https://pypi.org/pypi/python-dateutil/2.9.0.post0/json) |
| `pytz` | `2026.5` | MIT | [2026.5](https://pypi.org/pypi/pytz/2026.5/json) |
| `pyyaml` | `6.0.3` | MIT | [6.0.3](https://pypi.org/pypi/pyyaml/6.0.3/json) |
| `regex` | `2024.11.6` | Apache Software License | [2024.11.6](https://pypi.org/pypi/regex/2024.11.6/json) |
| `requests` | `2.34.2` | Apache-2.0 | [2.34.2](https://pypi.org/pypi/requests/2.34.2/json) |
| `safetensors` | `0.5.3` | Apache Software License | [0.5.3](https://pypi.org/pypi/safetensors/0.5.3/json) |
| `sentencepiece` | `0.2.0` | Apache | [0.2.0](https://pypi.org/pypi/sentencepiece/0.2.0/json) |
| `setuptools` | `75.8.0` | MIT License | [75.8.0](https://pypi.org/pypi/setuptools/75.8.0/json) |
| `six` | `1.17.0` | MIT | [1.17.0](https://pypi.org/pypi/six/1.17.0/json) |
| `sympy` | `1.13.1` | BSD | [1.13.1](https://pypi.org/pypi/sympy/1.13.1/json) |
| `tokenizers` | `0.21.1` | Apache Software License | [0.21.1](https://pypi.org/pypi/tokenizers/0.21.1/json) |
| `torch` | `2.6.0` | BSD-3-Clause | [2.6.0](https://pypi.org/pypi/torch/2.6.0/json) |
| `torch` | `2.6.0+cpu`, `2.6.0+cu124` | BSD-3-Clause; base-release metadata only* | [2.6.0](https://pypi.org/pypi/torch/2.6.0/json) |
| `torchvision` | `0.21.0` | BSD | [0.21.0](https://pypi.org/pypi/torchvision/0.21.0/json) |
| `torchvision` | `0.21.0+cpu`, `0.21.0+cu124` | BSD; base-release metadata only* | [0.21.0](https://pypi.org/pypi/torchvision/0.21.0/json) |
| `tqdm` | `4.67.1` | MPL-2.0 AND MIT | [4.67.1](https://pypi.org/pypi/tqdm/4.67.1/json) |
| `transformers` | `4.49.0` | Apache 2.0 License | [4.49.0](https://pypi.org/pypi/transformers/4.49.0/json) |
| `typing-extensions` | `4.12.2` | Python Software Foundation License | [4.12.2](https://pypi.org/pypi/typing-extensions/4.12.2/json) |
| `tzdata` | `2026.5` | Apache-2.0 | [2026.5](https://pypi.org/pypi/tzdata/2026.5/json) |
| `urllib3` | `2.8.0` | MIT | [2.8.0](https://pypi.org/pypi/urllib3/2.8.0/json) |
| `wcwidth` | `0.9.2` | MIT License | [0.9.2](https://pypi.org/pypi/wcwidth/0.9.2/json) |
| `wheel` | `0.45.1` | MIT License | [0.45.1](https://pypi.org/pypi/wheel/0.45.1/json) |

\* PyTorch `+cpu` / `+cu...` wheels come from the PyTorch wheel indexes, not these base PyPI releases. The corresponding base release metadata is shown only for the Python project; this does not verify the exact wheel's bundled libraries or their licenses. Consult the [PyTorch wheel index](https://download.pytorch.org/whl/) and the license/notice files shipped in the chosen wheel.
