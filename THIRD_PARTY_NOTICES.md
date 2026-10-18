# Third-party licenses and acknowledgments

AffordanceWAM's original contributions are distributed under the [MIT License](LICENSE).
Third-party code, packages, model assets, datasets, and tools retain their own
licenses. The project license does not replace those terms or grant rights to
third-party weights, data, or trademarks.

## Bundled and adapted source code

The [`affordance_wam/wan_vendor/`](affordance_wam/wan_vendor/) tree contains
adapted video-backbone and support code. It is a local integration, not an
unmodified upstream distribution. Integration changes include module extraction,
local model loading, lazy dependency imports, and attention handling. Existing
source attributions and the original vendor license are retained.

| Upstream project | Scope in this repository | License and retained text |
| --- | --- | --- |
| [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Wan backbone, model loading, diffusion, device, VRAM, and shared support modules in `affordance_wam/wan_vendor/`; explicitly identified by the existing vendor notice | Apache-2.0; [existing LICENSE](affordance_wam/wan_vendor/license/LICENSE) and [NOTICE](affordance_wam/wan_vendor/license/NOTICE). The license includes `Copyright [2023] [Zhongjie Duan]`. |
| [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Wan2.1](https://github.com/Wan-Video/Wan2.1) | Wan model lineage used by the DiffSynth integration, including text/image encoder and VAE components; Wan2.2-TI2V-5B is the configured backbone, while the vendor tree also retains Wan2.1-compatible components | Apache-2.0; [retained license](licenses/Wan-LICENSE.txt); upstream [Wan2.2 license](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt) and [Wan2.1 license](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt). |
| [CameraCtrl](https://github.com/hehao13/CameraCtrl) | `Camera`, `get_relative_pose`, and `ray_condition` in [`camera_controller.py`](affordance_wam/wan_vendor/camera_controller.py) explicitly identify CameraCtrl as their source | Apache-2.0; [retained license](licenses/CameraCtrl-LICENSE.txt); [upstream source](https://github.com/hehao13/CameraCtrl/blob/main/inference.py). |
| [OpenAI CLIP](https://github.com/openai/CLIP) | [`models/image_encoder.py`](affordance_wam/wan_vendor/models/image_encoder.py) identifies CLIP as an implementation source | MIT; [retained license and OpenAI copyright](licenses/OpenAI-CLIP-LICENSE.txt). |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | `models/image_encoder.py` also identifies OpenCLIP as an implementation source | MIT; [retained license and author notices](licenses/OpenCLIP-LICENSE.txt). This is the actual upstream license text, irrespective of automated license-detector labels. |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | `models/image_encoder.py` identifies its XLM-RoBERTa helper as adapted from Hugging Face; the Wan text-encoder lineage also identifies Transformers' T5 implementation | Apache-2.0; [retained license](licenses/Transformers-LICENSE.txt) and the source attributions below. Transformers is also installed separately as a runtime dependency. |

Additional upstream source attributions are retained here for the corresponding
encoder lineage:

- Wan: `Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.`
  See upstream [Wan T5](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/t5.py)
  and [Wan CLIP](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/clip.py).
- Transformers T5: `Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.`
  See [T5 source](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/t5/modeling_t5.py).
- Transformers XLM-RoBERTa: `Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.`
  and `Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.`
  See [XLM-RoBERTa source](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py).

The release's vendor snapshot does not record the exact original upstream source
commit. Links and license snapshots identify the upstream projects and verified
license texts; they do not assert that the local code was copied from today's
upstream revision or from the Transformers reference version above.

## Separately installed software

Declared Python packages and their published license metadata are listed in the
[Python dependency inventory](licenses/PYTHON_DEPENDENCIES.md). These packages are
installed by `pip`; their distributions are not included in this source repository.
Retain each installed distribution's own license and third-party notices when
redistributing an environment, wheel bundle, container, or executable.

The following software is also referenced by documentation or optional imports:

| Software | Relationship | Upstream license / terms |
| --- | --- | --- |
| [frankapy](https://github.com/iamlab-cmu/frankapy) | Real-world robot control software credited in the README; obtain and install separately | MIT; [retained license](licenses/frankapy-LICENSE.txt), `Copyright (c) 2021 Carnegie Mellon University`. Its underlying robot libraries and SDKs retain their own terms. |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Optional `flash_attn` / `flash_attn_interface` imports in vendor attention code; the documented setup uses PyTorch SDPA | [BSD-3-Clause](https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE). |
| [SageAttention](https://github.com/thu-ml/SageAttention) | Optional `sageattention` backend | [Apache-2.0](https://github.com/thu-ml/SageAttention/blob/main/LICENSE). |
| [xFormers](https://github.com/facebookresearch/xformers) | Optional backend in shared attention utilities | [BSD-style license and upstream notices](https://github.com/facebookresearch/xformers/blob/main/LICENSE). |
| [Ascend Extension for PyTorch](https://github.com/Ascend/pytorch) | Optional `torch_npu` support in vendor device utilities | [Upstream BSD-style license and attribution notices](https://github.com/Ascend/pytorch/blob/master/LICENSE); Ascend runtime components have separate terms. |
| [ModelScope](https://github.com/modelscope/modelscope) | Optional model-download SDK in inherited loader utilities; the standard AffordanceWAM path loads local model files | [Apache-2.0](https://github.com/modelscope/modelscope/blob/master/LICENSE). SDK licensing does not license the models downloaded through it. |
| [FFmpeg](https://ffmpeg.org/) | Video encoding/decoding through PyAV and imageio-ffmpeg, including the `libx264` encoding check in `env/check.py` | [FFmpeg license information](https://ffmpeg.org/legal.html). FFmpeg is primarily LGPL-2.1-or-later; enabling GPL components such as x264 changes the applicable build terms. Inspect the actual binary's configuration and bundled notices before redistribution. Python wrapper licenses do not replace codec/binary licenses. |
| [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) | NVIDIA profile downloads NVCC 12.4.131 in `env/install_nvcc.py`; CUDA runtime libraries arrive with the selected PyTorch installation | [CUDA 12.4.1 EULA and CUDA supplement](https://docs.nvidia.com/cuda/archive/12.4.1/eula/index.html). These separately downloaded NVIDIA components are not covered by this project's MIT license. |
| [Miniforge](https://github.com/conda-forge/miniforge) | Optional Python bootstrap described in `env/README.md` | Installer code: [BSD-3-Clause](https://github.com/conda-forge/miniforge/blob/main/LICENSE); bundled and installed packages retain their own licenses. |

Optional backend references describe available integration points, not a claim
that every backend is installed, required, or supported by the default environment.

## Model weights and tokenizers

Model weights and tokenizer assets are downloaded separately and are not included
in this code release.

- **Wan2.2-TI2V-5B:** the [official model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
  declares Apache-2.0. Preserve the model's own license and notices when sharing
  its weights or derivatives.
- **UMT5-XXL:** the configured tokenizer path is `google/umt5-xxl`, and the World
  Expert uses the UMT5 text encoder. The [Google model card](https://huggingface.co/google/umt5-xxl)
  declares Apache-2.0. Checkpoint export may copy tokenizer files through
  `component_specs.py`; include their applicable license/attribution when sharing
  such an exported checkpoint.
- **Converted / alternative Wan assets:** `loader.py` retains redirects to
  [DiffSynth-Studio/Wan-Series-Converted-Safetensors](https://huggingface.co/DiffSynth-Studio/Wan-Series-Converted-Safetensors),
  and the vendor registry contains other model identifiers. These references are
  not a blanket license grant for every listed model. Follow the original model
  publisher's license and the exact downloaded repository's model card; a format
  conversion does not replace the original terms.

## Benchmarks, datasets, and related research

The README describes evaluation with [RoboCasa](https://github.com/robocasa/robocasa)
and [CALVIN](https://github.com/mees/calvin). Their benchmark software is obtained
separately: see the [RoboCasa license](https://github.com/robocasa/robocasa/blob/main/LICENSE)
(MIT with additional MuJoCo attribution) and [CALVIN license](https://github.com/mees/calvin/blob/main/LICENSE)
(MIT). Simulator assets, demonstrations, and benchmark datasets can have terms
separate from the benchmark code license.

The human and robot data sources listed in [data/README.md](data/README.md),
including Ego4D-FHO, EPIC-KITCHENS VISOR, HOI4D, H2O, MECCANO, RoboInter/DROID/RH20T,
BridgeData V2, InternData-A1, and RoboCasa365, are not included in this code release.
Obtain them from their publishers and follow the specific dataset licenses,
access agreements, attribution requirements, and any use or redistribution
restrictions. This repository does not relicense those datasets.

AffordanceVLA, A3D, PA3FF, DexGarmentLab, and DualAfford in the README's related
projects section are research links. A citation or link does not by itself mean
that their code is bundled here or licensed under AffordanceWAM's MIT license.

License references were checked on **2026-10-18**. See the [license snapshot index](licenses/README.md)
for the source and checksum of each retained license text. This notice records
attribution and scope; the full applicable upstream license texts control.
