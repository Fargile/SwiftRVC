# Third-Party Licenses & Acknowledgments

SwiftRVC's own source code (`app/`, `engine/`, `beatrice/beatrice_bridge.cpp`, `setup.sh`, `build.sh`) is licensed under the MIT License.

Optional downloaded weights, libraries, and models fetched by `setup.sh` or supplied by the user remain subject to their respective upstream licenses:

- **Apple MLX** (`mlx`, `mlx-metal`) — MIT License — https://github.com/ml-explore/mlx
- **MLX ContentVec & RMVPE weights** (`lexandstuff/mlx-contentvec`, `lexandstuff/mlx-rmvpe`) — MIT License — Hugging Face
- **Retrieval-based Voice Conversion (RVC v2)** — MIT License — https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- **Project Beatrice (`libbeatrice.a` / `beatrice.h` / Beatrice v2 models)** — Provided separately by Project Beatrice (`prj-beatrice/beatrice-vst`); precompiled binaries and JVS/voice datasets are not bundled in this repository and follow Project Beatrice's upstream license terms.
- **BlackHole 2ch** (optional virtual audio driver for Discord routing) — GPLv3 — https://github.com/ExistentialAudio/BlackHole
