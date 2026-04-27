# OldVideo HDRemasterer AI

AI アップスケーリング (Real-ESRGAN) とフレーム補間 (RIFE)、および実写向け補正機能を統合した動画リマスターツールです。 / 
A powerful video remastering tool that integrates AI upscaling (Real-ESRGAN), frame interpolation (RIFE), and stabilization for live-action footage.

## 主要な機能 / Key Features

- **GUI ダッシュボード / GUI Dashboard**: 直感的なウィンドウ操作でリマスターが可能。 / Remaster videos with an intuitive window interface.
- **フレーム補間 (RIFE) / Frame Interpolation (RIFE)**: 24/30fps の動画を 60fps などへ滑らかに変換。 / Smoothly convert 24/30fps videos to 60fps or more.
- **手ブレ補正 / Video Stabilization**: OpenCV による高度な安定化処理。 / Advanced stabilization processing using OpenCV.
- **マルチ言語対応 / Multilingual Support**: 日本語と英語の表示切替に対応。 / Supports switching between Japanese and English.

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

## ライセンス / License
MIT License.
