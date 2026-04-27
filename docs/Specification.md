# OldVideo HDRemasterer AI 技術仕様書 / Technical Specifications

## 1. 概要 / Overview
本ソフトウェアは、低解像度の古い動画を現代の 4K または HD 解像度にアップスケーリングし、映像品質を向上させるための C++ ツールです。 / 
This software is a C++ tool designed to upscale low-resolution legacy videos to modern 4K or HD resolutions and improve video quality.

## 2. 技術スタック / Technology Stack
- **言語 / Language**: C++17
- **ビルド / Build**: CMake (3.10+)
- **映像エンジン / Engine**:
    - **OpenCV 4.11.0**: [https://opencv.org/](https://opencv.org/) - Stabilization, Encoding/Decoding.
    - **ncnn**: [https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn) - Deep Learning inference (Vulkan GPU).
    - **RIFE v2/v3/v4**: [https://github.com/hzwer/arXiv2020-RIFE](https://github.com/hzwer/arXiv2020-RIFE) - AI frame interpolation.
    - **Real-ESRGAN**: [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI super-resolution.
    - **FFmpeg**: [https://ffmpeg.org/](https://ffmpeg.org/) - Audio merging and hardware encoding control.

## 3. 主要な機能 / Key Features

### 3.1. インテリジェント GPU 選択 / Intelligent GPU Selection
システムに搭載された最適な GPU（Vulkan 対応）を自動的に優先選択します。 / 
Automatically selects the best available GPU (Vulkan-compatible) on the system.

### 3.2. 最適化済みパイプライン / Optimized Pipeline
AI モデルがノイズ処理を兼ねるため、不要な前処理を排除し CPU 負荷を軽減しています。 / 
Reduces CPU load by eliminating redundant preprocessing steps as AI models handle noise implicitly.

### 3.3. タイル処理 (VRAM 最適化) / Tiling (VRAM Optimization)
GPU の利用可能メモリに応じてタイルサイズを自動調整し、大解像度処理時のクラッシュを防止します。 / 
Prevents crashes during high-resolution processing by auto-adjusting tile sizes based on VRAM.

### 3.4. 多言語対応 (New) / Multilingual Support (New)
UI とドキュメントの日本語・英語切替に対応しました。 / 
Added support for switching between Japanese and English in UI and documentation.

## 4. ユーザーインターフェース / User Interface
- **Dashboard**: Dear ImGui ベースのリアルタイム進捗監視。 / Dear ImGui-based real-time progress monitoring.
- **Progress Bar**: 処理の進行状況をパーセント表示。 / Displays processing progress in percentages.
- **Language Switch**: JP / EN の即時切替が可能。 / Immediate switching between JP and EN.

## 5. ファイル構成 / File Structure (v2.1+)

```
OldVideoHDRemasterer_CPP/
├── CMakeLists.txt              # ビルド設定 / Build config
├── README.md                   # 説明書 / Instructions
├── .gitignore                  # 除外設定 / Git exclusion rules
├── docs/                       # ドキュメント / Documentation
│   └── Specification.md        # 本仕様書 / This specification
├── scripts/                    # スクリプト / Scripts
│   └── download_models.ps1     # モデル取得 / Model downloader
├── src/                        # ソースコード / Source code
│   ├── main.cpp                # Entry point
│   ├── gui.cpp / gui.h         # ImGui Dashboard
│   ├── localization.h          # 多言語化 / Localization
│   └── ...
└── models/                     # AI モデル / AI Models
```

## 6. ライセンス / License
MIT License / BSD 3-Clause (ncnn) / Apache 2 (OpenCV).
