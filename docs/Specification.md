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

### 3.2. 堅牢なスレッド安全性 (New) / Robust Thread Safety (New)
GUI スレッドと動画処理スレッドを完全に分離し、Mutex による排他制御を実装。データレースによるクラッシュを防止しています。 / 
Completely separates GUI and video processing threads with Mutex-based synchronization to prevent data-race crashes.

### 3.3. ユニコード・パス対応 (New) / Unicode Path Support (New)
日本語などの全角文字を含むファイルパスを正しく処理できるよう、Windows UTF-16 と内部 UTF-8 の変換を最適化。日本語 Windows 環境での安定性を確保しました。 / 
Optimized conversion between Windows UTF-16 and internal UTF-8 for correct handling of Japanese and other Unicode file paths.

### 3.4. 自動 CPU フォールバック / Automatic CPU Fallback
VRAM 不足や GPU エラーを検知した際、自動的に CPU 処理へ切り替えることで処理の完遂を保証します。 / 
Guarantees completion by automatically falling back to CPU processing when VRAM is insufficient or GPU errors occur.

### 3.5. 中断ボタンの実装 (New) / Stop Functionality (New)
処理中に安全にリマスターを停止し、それまでの進捗を破棄せずに終了できる中断ボタンを実装しました。 / 
Added a stop button to safely terminate the process without crashing the application.

## 4. ユーザーインターフェース / User Interface
- **Dashboard**: Dear ImGui ベースのリアルタイム進捗監視。 / Dear ImGui-based real-time progress monitoring.
- **Progress Bar**: 処理の進行状況をパーセント表示。 / Displays processing progress in percentages.
- **Diagnostics**: 異常終了時に原因を特定できるよう、Windows SEH ハンドラと詳細なログ出力を搭載。 / Equipped with Windows SEH handler and detailed logging for crash diagnostics.

## 5. ファイル構成 / File Structure (v2.2+)

```
OldVideoHDRemasterer_CPP/
├── CMakeLists.txt              # ビルド設定 / Build config
├── README.md                   # 説明書 / Instructions
├── docs/                       # ドキュメント / Documentation
│   └── Specification.md        # 本仕様書 / This specification
├── src/                        # ソースコード / Source code
│   ├── main.cpp                # Entry point (ncnn init/log setup)
│   ├── gui.cpp / gui.h         # ImGui Dashboard (Safe thread handling)
│   ├── video_processor.cpp     # Processing pipeline (Stop flag logic)
│   ├── localization.h          # 多言語化 / Localization
│   └── ...
└── models/                     # AI モデル / AI Models
```

## 6. 既知の修正事項 / Latest Fixes
- **GPU 初期化の安定化**: アプリケーション起動時に Vulkan インスタンスを明示的に作成。
- **データレース修正**: 進捗表示用文字列への同時アクセスを Mutex で保護。
- **文字コード不整合の解消**: `std::filesystem` で日本語パスを扱う際の例外エラーを修正。

## 7. 推奨される入力動画の仕様 / Recommended Input Specifications
高品質なリマスター結果を得るためのガイドラインです。

- **推奨解像度**: 480p (720x480) 以上を推奨。360p でも良好な結果が得られます。
- **ビットレート**: 高いほど AI が正確なディテールを復元できます。
- **フレームレート**: 24/30fps の素材を RIFE で 60fps 化するのが最も一般的です。
- **ノイズ**: 圧縮ノイズが激しい素材は、AI がノイズを強調する可能性があるため、低圧縮の素材が望ましいです。

## 8. ライセンス / License
MIT License / BSD 3-Clause (ncnn) / Apache 2 (OpenCV).
