<p align="center">
  <img src="assets/logo.png" width="160" height="160" alt="OldVideo HDRemasterer AI Icon">
</p>

# OldVideo HDRemasterer AI

AI アップスケーリング (Real-ESRGAN) とフレーム補間 (RIFE)、および実写向け補正機能を統合した動画リマスターツールです。 / 
A powerful video remastering tool that integrates AI upscaling (Real-ESRGAN), frame interpolation (RIFE), and stabilization for live-action footage.

## 主要な機能 / Key Features

- **GUI ダッシュボード / GUI Dashboard**: 直感的なウィンドウ操作でリマスターが可能。 / Remaster videos with an intuitive window interface.
- **フレーム補間 (RIFE) / Frame Interpolation (RIFE)**: 24/30fps の動画を 60fps などへ滑らかに変換。 / Smoothly convert 24/30fps videos to 60fps or more.
- **手ブレ補正 / Video Stabilization**: OpenCV による高度な安定化処理。 / Advanced stabilization processing using OpenCV.
- **マルチ言語対応 / Multilingual Support**: 日本語と英語の表示切替に対応。 / Supports switching between Japanese and English.
- **バッチ処理 / Batch Processing**: 複数の動画タスクを一括登録し、連続自動リマスターが可能。 / Queue and process multiple video files sequentially.
- **比較プレビュー / Before-After Split Preview**: 境界をスライドできるスリットスライダーで、画質向上効果をリアルタイムで直接比較可能。 / Real-time split-screen slider to compare before and after remastered frames directly.
- **部分処理 / Trim & Preview**: 開始位置と秒数を指定して、特定区間のみの部分テスト処理が可能。 / Remaster a specific duration (start/length) for fast previewing.
- **FFmpeg 自動配置 / Auto FFmpeg Installer**: 音声マージに不可欠な FFmpeg をワンクリックで自動ダウンロード・配置。 / One-click tool to auto-download and configure FFmpeg.
- **GPU+CPU ハイブリッド処理 / GPU+CPU Hybrid Processing**: AI超解像をGPUで、フレーム補間をCPUで動作させ、低VRAM（4GB等）環境でのクラッシュを防止。 / Runs AI upscaling on GPU and frame interpolation on CPU to prevent crashes in low-VRAM (e.g. 4GB) environments.
- **セキュリティと堅牢性 / Security & Robustness**: コマンドインジェクション対策のサニタイズ強化、Use-After-Free脆弱性の解消、NaNデータ安全フォールバックの搭載。 / Enhanced shell path sanitization, fixed Use-After-Free thread issues, and added NaN/Inf safety fallbacks for maximum system robustness.

## 推奨される入力動画の仕様 / Recommended Input Specifications

リマスターの効果を最大限に引き出すための推奨条件です。 / Recommended conditions for best results.

- **解像度 / Resolution**: 
  - **Best**: 480p (640x480, 720x480) - DVD 画質
  - **Good**: 360p (640x360)
  - **Notice**: 240p 以下は補完が不自然になる場合があります。 / Resolutions below 240p may result in artifacts.
- **品質 / Quality**: 
  - ブロックノイズが少ない動画ほど、AI はきれいにディテールを復元できます。 / Less compression noise allows for better AI restoration.

## 必要条件 / Requirements

- **OS**: Windows (Vulkan 対応の GPU が必須 / Vulkan-compatible GPU required)
- **Tools**:
  - CMake (3.10+)
  - C++ Compiler (Visual Studio 2019/2022 recommended)
  - OpenCV (videostab module required)
  - FFmpeg (Must be in PATH)

## セットアップ手順 / Setup Instructions

### 1. ビルド環境の構築 / Build Environment Setup

依存ライブラリは `vcpkg` で管理することを推奨します。 / We recommend using `vcpkg` to manage dependencies.

```powershell
# vcpkg で必要なライブラリをインストール / Install required libraries via vcpkg
vcpkg install opencv4[ffmpeg,videostab,world] ncnn cxxopts vulkan glfw3 imgui[opengl3-binding,glfw-binding] --triplet x64-windows
```

### 2. モデルのダウンロード / Downloading Models

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1
```

### 3. ビルド / Build

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[vcpkg_path]/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

## 使い方 / Usage

### GUI モード (推奨) / GUI Mode (Recommended)
引数なしで実行ファイルを起動してください。 / Run the executable without arguments.
```bash
./OldVideoHDRemasterer_CPP.exe
```

### CLI モード / CLI Mode
コマンドラインから自動化を行う場合に使用します。 / Use for automation from the command line.
```bash
./OldVideoHDRemasterer_CPP --input input.mp4 --output output.mp4 --scale 4 --interp --stab
```

## 外部ソフトウェアとライブラリ / External Software & Libraries

本プロジェクトでは、以下の優れたオープンソースソフトウェアおよびライブラリを使用しています。 / This project utilizes the following excellent open-source software and libraries.

- **FFmpeg**: [https://ffmpeg.org/](https://ffmpeg.org/) - 映像・音声のエンコードとデコード / Video & Audio encoding/decoding.
- **OpenCV**: [https://opencv.org/](https://opencv.org/) - 映像解析・手ブレ補正 / Computer vision & stabilization.
- **ncnn**: [https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn) - 高性能推論エンジン / High-performance neural network inference.
- **Real-ESRGAN**: [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 超解像 AI モデル / Super-resolution AI model.
- **RIFE**: [https://github.com/hzwer/arXiv2020-RIFE](https://github.com/hzwer/arXiv2020-RIFE) - フレーム補間 AI モデル / Frame interpolation AI model.
- **Dear ImGui**: [https://github.com/ocornut/imgui](https://github.com/ocornut/imgui) - グラフィカル UI / Graphical UI.
- **Vulkan**: [https://www.vulkan.org/](https://www.vulkan.org/) - 高速な GPU アクセラレーション / High-performance GPU acceleration.

## ライセンス / License
MIT License.
