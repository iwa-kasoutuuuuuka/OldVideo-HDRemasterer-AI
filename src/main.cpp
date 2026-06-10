#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#include <iostream>
#include <cstdio>
#include <locale>
#include <cxxopts.hpp>
#include "gpu.h"
#include "video_processor.h"
#include "gui.h"

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); // コンソールの出力を UTF-8 に設定
    std::locale::global(std::locale(".UTF8"));
#endif

    // ncnn GPU インスタンスの初期化
    try {
        ncnn::create_gpu_instance();
    } catch (const std::exception& e) {
        std::cerr << "[CRASH] ncnn 初期化エラー: " << e.what() << std::endl << std::flush;
        return -1;
    }

    // 自動ログ出力の設定 (上書き競合を防ぐため、stdout を freopen した後 _dup2 で stderr に複製する)
    {
        FILE* tmp = std::fopen("latest_log.txt", "w");
        if (tmp) std::fclose(tmp);
    }
    FILE* f_out = std::freopen("latest_log.txt", "a", stdout);
    if (f_out) {
#ifdef _WIN32
        _dup2(_fileno(stdout), _fileno(stderr));
#else
        dup2(fileno(stdout), fileno(stderr));
#endif
    } else {
        // ログファイルを開けない場合でも処理は続行
        std::cerr << "[WARNING] ログファイルのオープンに失敗しました。" << std::endl;
    }
    std::setvbuf(stdout, NULL, _IONBF, 0);
    std::setvbuf(stderr, NULL, _IONBF, 0);

    std::cout << "--- OldVideo HDRemasterer AI - Session Started ---" << std::endl;
    std::cout << "--- 動画リマスターセッション開始 ---" << std::endl;

    // Priority: Run GUI if no arguments are provided
    if (argc <= 1) {
        std::cout << "[INFO] 引数が指定されていません。GUIモードを開始します... / No arguments provided. Starting GUI mode..." << std::endl;
        int gui_ret = GUIManager::run();
        ncnn::destroy_gpu_instance();
        return gui_ret;
    }

    try {
        cxxopts::Options options("OldVideo HDRemasterer AI", "Real-ESRGAN と OpenCV に基づく AI 動画アップスケーラー / AI Video Upscaler based on Real-ESRGAN and OpenCV");

        options.add_options()
            ("i,input", "入力動画のパス (必須) / Input video path (Required)", cxxopts::value<std::string>())
            ("o,output", "出力動画の保存パス (必須) / Output video path (Required)", cxxopts::value<std::string>())
            ("s,scale", "拡大率 (2 または 4) / Upscale ratio (2 or 4)", cxxopts::value<int>()->default_value("4"))
            ("t,tile", "VRAM節約のためのタイルサイズ (0 で自動) / Tile size for VRAM saving (0 for auto)", cxxopts::value<int>()->default_value("0"))
            ("m,models", "モデルディレクトリのパス / Path to models directory", cxxopts::value<std::string>()->default_value("models"))
            ("stab", "手ブレ補正を有効化 / Enable video stabilization", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("interp", "フレーム補間 (RIFE) を有効化 / Enable frame interpolation (RIFE)", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("model_name", "モデル名 (例: realesr-animevideov3-x2) / Specific model name", cxxopts::value<std::string>()->default_value(""))
            ("trim_start", "トリミング開始位置 (秒) / Trim start position in seconds", cxxopts::value<double>()->default_value("0.0"))
            ("trim_duration", "トリミング時間 (秒, 0で無効) / Trim duration in seconds (0 to disable)", cxxopts::value<double>()->default_value("0.0"))
            ("h,help", "ヘルプを表示 / Show help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        if (!result.count("input") || !result.count("output")) {
            std::cerr << "[ERROR] 入力パスと出力パスは必須です。 / Input and output paths are required." << std::endl;
            std::cout << options.help() << std::endl;
            return -1;
        }

        VideoProcessor::Config config;
        config.input_path = result["input"].as<std::string>();
        config.output_path = result["output"].as<std::string>();
        config.scale = result["scale"].as<int>();
        config.tile_size = result["tile"].as<int>();
        config.model_dir = result["models"].as<std::string>();
        config.model_name = result["model_name"].as<std::string>();
        config.enable_stabilization = result["stab"].as<bool>();
        config.enable_interpolation = result["interp"].as<bool>();

        double trim_start = result["trim_start"].as<double>();
        double trim_dur = result["trim_duration"].as<double>();
        if (trim_dur > 0.0) {
            config.enable_trim = true;
            config.trim_start_sec = trim_start;
            config.trim_duration_sec = trim_dur;
        } else {
            config.enable_trim = false;
        }

        if (config.scale != 2 && config.scale != 4) {
            std::cerr << "[ERROR] 拡大率は 2 または 4 である必要があります。 / Scale must be 2 or 4." << std::endl;
            return -1;
        }

        VideoProcessor processor;
        if (processor.run(config)) {
            std::cout << "[SUCCESS] 完了しました。 / Completed successfully." << std::endl;
            return 0;
        } else {
            std::cerr << "[ERROR] 処理中にエラーが発生しました。 / An error occurred during processing." << std::endl;
            return 1;
        }

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "[ERROR] 引数の解析エラー: " << e.what() << " / Argument parsing error." << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] 予期せぬエラーが発生しました: " << e.what() << " / An unexpected error occurred." << std::endl;
        ncnn::destroy_gpu_instance();
        return -1;
    }

    ncnn::destroy_gpu_instance();
}
