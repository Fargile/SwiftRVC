# SwiftRVC

Lightweight native macOS real-time voice changer for Apple Silicon (built to run cleanly on base **M1 8GB** Macs) supporting **RVC v2 (Apple MLX)** and **Beatrice v2 (ARM-NEON)**.

No Electron, no PyTorch, no FAISS, no Xcode required.

## Features

- **RVC v2 on Apple MLX:** Zero-PyTorch `.pth` loader, memory-mapped `.index` (~6 MB RAM instead of 270+ MB), BN-folded RMVPE + Fast YIN, tail-cropped HiFi-GAN, SOLA crossfade, 5 speed/quality presets (`~390–940 MB` RAM, `0% GPU` when gated).
- **Beatrice v2 (optional, native ARM-NEON):** Runs via `libbeatrice.a` (`2.0.0-alpha.2`, `beta.1`, `rc.0`), ~60–125 ms total latency, ~150 MB RAM, `0% GPU`. Switching engines unloads the inactive engine completely.
- **3 Live Input Sources:**
  - **Mic:** Native CoreAudio capture with cubic resampling (44.1 kHz → 48 kHz).
  - **File:** Load `.wav`/`.m4a`/`.mp3`/`.flac`/`.ogg`/`.mp4` (or record a quick mic sample), loop/seek live while tuning, or offline **Export WAV**.
  - **App (macOS 14.2+):** Native CoreAudio Process Tap for System Audio or a specific app, with **Mute original** toggle.
- **Built-in Live Recorder (`Rec`):** Streams converted 48 kHz mono WAV output straight to `recordings/` with inline playback preview, **Save**, and **Finder** reveal.
- **Voice Tuning:** Pitch (`-24..+24 st`), Formant shift, soft Pitch Floor, Polish (leveler, de-bass, F0 smoothing, EQ), Gate + In/Out meters, Index blend, Protect / Shimmer / Intonation / RMS Mix / Gain, auto-saved per model in `model_settings.json`.

## Performance (M1 8GB)

| Engine / Preset | Inference | Total Latency | RAM |
|---|---:|---:|---:|
| RVC v2 · Ultra Perf (8-bit + YIN) | ~35 ms | ~125 ms | ~390–500 MB |
| RVC v2 · Balanced (RMVPE fp16) | ~75–95 ms | ~270 ms | ~750 MB |
| RVC v2 · Quality / Ultra Qual (RMVPE + mmap index) | ~125–170 ms | ~420–575 ms | ~850–940 MB |
| Beatrice v2 · Balanced (CPU NEON) | ~5–8 ms | ~82 ms | ~155 MB (0% GPU) |

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4), macOS 13+ (macOS 14.2+ for App audio capture)
- Xcode Command Line Tools (`xcode-select --install`) — full Xcode is not needed
- Python 3.11 or 3.12
- Optional: [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) (`brew install blackhole-2ch`) for Discord routing, and `ffmpeg` (`brew install ffmpeg`) for decoding `.mp4`/`.ogg`/`.webm` files

## Install & Build

```bash
git clone <your-repo-url>
cd SwiftRVC

./setup.sh   # creates .venv, installs mlx/numpy/scipy, downloads MLX base weights
./build.sh   # compiles SwiftRVC.app (no Xcode needed)
open SwiftRVC.app
```

### Adding Voice Models

- **RVC v2:** Click **`+ Add`** in the app (or drop `.pth` + optional `.index` into `models/<name>/`).
- **Beatrice v2 (optional):** Place macOS arm64 `libbeatrice.a` and `beatrice.h` from [`prj-beatrice/beatrice-vst`](https://github.com/prj-beatrice/beatrice-vst) into `beatrice/lib/`, re-run `./setup.sh`, then add any Beatrice v2 `.zip` or `.toml` pack via **`+ Add`** (or drop into `beatrice_models/`).

### Discord Setup

1. Install BlackHole 2ch (`brew install blackhole-2ch`) and set **Output** in SwiftRVC to `BlackHole 2ch`.
2. In **Discord → Settings → Voice & Video**, set **Input Device** to `BlackHole 2ch` and turn **Noise Suppression** off.
3. For App/System capture on macOS 14.2+, allow SwiftRVC under **System Settings → Privacy & Security → Screen & System Audio Recording → System Audio Recording Only**.

## License

MIT — see [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
