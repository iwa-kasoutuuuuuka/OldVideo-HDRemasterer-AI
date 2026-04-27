#include "video_processor.h"
#include "utils.h"
#include "stabilizer.h"
#include "interpolator.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ncnn/gpu.h>

namespace fs = std::filesystem;

int VideoProcessor::get_auto_tile_size() {
    if (ncnn::get_gpu_count() == 0) return 256; // Fallback for CPU
    uint32_t budget_mb = ncnn::get_gpu_device(0)->get_heap_budget();
    
    // Automatically clamp tile size based on approx total VRAM budget
    if (budget_mb >= 8000) return 1024;
    if (budget_mb >= 4000) return 512;
    if (budget_mb >= 2000) return 256;
    return 128;
}

VideoProcessor::VideoProcessor() {}

bool VideoProcessor::run(const Config& config) {
    // 1. 各種コンポーネントの初期化
    Stabilizer stabilizer_svc;
    Interpolator interpolator_svc;
    
    cv::VideoCapture cap;
    bool use_stabilizer = config.enable_stabilization;

    if (use_stabilizer) {
        std::cout << "[INFO] 手ブレ補正を有効化しました。" << std::endl;
        if (!stabilizer_svc.init(config.input_path)) {
            return false;
        }
    } else {
        cap.open(config.input_path);
        if (!cap.isOpened()) {
            std::cerr << "[ERROR] 動画ファイルを開けません: " << config.input_path << std::endl;
            return false;
        }
    }

    // 基本情報の取得
    double fps = use_stabilizer ? stabilizer_svc.get_fps() : cap.get(cv::CAP_PROP_FPS);
    int total_frames = use_stabilizer ? stabilizer_svc.get_frame_count() : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int width = use_stabilizer ? 0 : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = use_stabilizer ? 0 : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 2. アップスケーラーと補完器のロード
    int actual_tile_size = config.tile_size;
    if (actual_tile_size <= 0) {
        actual_tile_size = get_auto_tile_size();
        std::cout << "[INFO] タイルサイズを自動設定しました: " << actual_tile_size << std::endl;
    }

    if (config.progress_callback) config.progress_callback(0.05f, "AIモデルのロード中...");
    if (!upscaler.load(config.model_dir, config.scale, actual_tile_size, config.use_fast_model)) {
        return false;
    }
    if (config.enable_interpolation) {
        if (!interpolator_svc.load(config.model_dir)) {
            return false;
        }
    }

    // 3. 出力設定
    std::string temp_video = "temp_no_audio.mp4";
    cv::VideoWriter writer;
    
    // 4. フレーム処理ループ (最適化済み: 不要コピー・後処理を削除)
    cv::Mat frame, upscaled, prev_upscaled;
    int current_frame = 0;

    std::cout << "[INFO] 処理を開始します..." << std::endl;
    if (config.progress_callback) config.progress_callback(0.1f, "フレーム処理中...");

    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        if (use_stabilizer) {
            frame = stabilizer_svc.next_frame();
        } else {
            cap.read(frame);
        }

        if (frame.empty()) break;

        // Writer の初期化 (初回のみ)
        if (!writer.isOpened()) {
            width = frame.cols;
            height = frame.rows;
            int target_width = width * config.scale;
            int target_height = height * config.scale;
            double output_fps = config.enable_interpolation ? fps * 2 : fps;
            writer.open(temp_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), output_fps, cv::Size(target_width, target_height));
        }

        // AI アップスケーリング (denoise/enhance はスキップ → AI に委任)
        if (!upscaler.process(frame, upscaled)) {
            std::cerr << "\n[ERROR] アップスケール失敗。" << std::endl;
            break;
        }

        // フレーム補間 (RIFE)
        if (config.enable_interpolation && !prev_upscaled.empty()) {
            cv::Mat interp_frame;
            if (interpolator_svc.process(prev_upscaled, upscaled, interp_frame)) {
                writer.write(interp_frame);
            }
        }

        // 本フレームの書き込み (後処理なしで直接出力)
        writer.write(upscaled);

        // clone() → swap() で高速化
        std::swap(prev_upscaled, upscaled);
        current_frame++;

        if (total_frames > 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
            double fps_processing = current_frame / elapsed_sec;
            double remaining_sec = (total_frames - current_frame) / std::max(fps_processing, 0.001);
            
            float p = 0.1f + 0.8f * ((float)current_frame / total_frames);
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1);
            ss << "処理中: " << current_frame << "/" << total_frames << " フレーム"
               << " | " << fps_processing << " fps | 残り時間: ";
               
            int r_h = static_cast<int>(remaining_sec) / 3600;
            int r_m = (static_cast<int>(remaining_sec) % 3600) / 60;
            int r_s = static_cast<int>(remaining_sec) % 60;
            ss << std::setfill('0') << std::setw(2) << r_h << ":"
               << std::setw(2) << r_m << ":" << std::setw(2) << r_s;
               
            if (config.progress_callback) config.progress_callback(p, ss.str());
        } else {
            std::cout << "\rProcessed frames: " << current_frame << std::flush;
        }
    }

    if (!use_stabilizer) cap.release();
    writer.release();

    // 5. 音声マージ
    if (config.progress_callback) config.progress_callback(0.95f, "音声のマージ中...");
    bool merge_ok = utils::merge_audio(temp_video, config.input_path, config.output_path);

    if (merge_ok && fs::exists(temp_video)) {
        fs::remove(temp_video);
    }

    if (config.progress_callback) config.progress_callback(1.0f, "完了しました！");
    return merge_ok;
}
