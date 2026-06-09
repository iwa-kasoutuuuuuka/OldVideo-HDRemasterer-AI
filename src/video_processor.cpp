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
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "gpu.h"

namespace fs = std::filesystem;

// スレッドセーフなキュー
template <typename T>
class SafeQueue {
private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond;
    size_t max_size;
    bool finished;

public:
    SafeQueue(size_t max_sz = 10) : max_size(max_sz), finished(false) {}

    void push(T value) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]() { return queue.size() < max_size || finished; });
        if (finished) return;
        queue.push(std::move(value));
        cond.notify_all();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]() { return !queue.empty() || finished; });
        if (queue.empty() && finished) return false;
        value = std::move(queue.front());
        queue.pop();
        cond.notify_all();
        return true;
    }

    void shutdown() {
        std::unique_lock<std::mutex> lock(mutex);
        finished = true;
        cond.notify_all();
    }

    size_t size() {
        std::unique_lock<std::mutex> lock(mutex);
        return queue.size();
    }
};

int VideoProcessor::get_auto_tile_size() {
    int gpu_idx = utils::get_preferred_gpu_index();
    if (gpu_idx < 0) return 256; // Fallback for CPU
    uint32_t budget_mb = ncnn::get_gpu_device(gpu_idx)->get_heap_budget();
    
    // Automatically clamp tile size based on approx total VRAM budget
    if (budget_mb >= 8000) return 400;
    if (budget_mb >= 4000) return 256;
    if (budget_mb >= 2000) return 128;
    return 64;
}

VideoProcessor::VideoProcessor() {}

bool VideoProcessor::run(const Config& config) {
    std::cout << "[RUN] Step 1: コンポーネント初期化中..." << std::endl << std::flush;
    
    // 1. 各種コンポーネントの初期化
    Stabilizer stabilizer_svc;
    Interpolator interpolator_svc;
    
    std::cout << "[RUN] Step 1: コンポーネント初期化完了" << std::endl << std::flush;
    
    cv::VideoCapture cap;
    bool use_stabilizer = config.enable_stabilization;

    if (use_stabilizer) {
        std::cout << "[INFO] 手ブレ補正を有効化しました。" << std::endl << std::flush;
        if (!stabilizer_svc.init(config.input_path)) {
            return false;
        }
    } else {
        std::cout << "[RUN] Step 2: 動画ファイルを開いています: " << config.input_path << std::endl << std::flush;
        cap.open(config.input_path);
        if (!cap.isOpened()) {
            std::cerr << "[ERROR] 動画ファイルを開けません: " << config.input_path << std::endl << std::flush;
            return false;
        }
        std::cout << "[RUN] Step 2: 動画ファイルを開きました" << std::endl << std::flush;
    }

    // 基本情報の取得
    double fps = use_stabilizer ? stabilizer_svc.get_fps() : cap.get(cv::CAP_PROP_FPS);
    int total_frames = use_stabilizer ? stabilizer_svc.get_frame_count() : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int width = use_stabilizer ? 0 : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = use_stabilizer ? 0 : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "[RUN] 動画情報: " << width << "x" << height << " @ " << fps << "fps, " << total_frames << " フレーム" << std::endl << std::flush;

    // 解像度バリデーション: 0x0 の場合は処理不可
    if (!use_stabilizer && (width <= 0 || height <= 0)) {
        std::cerr << "[ERROR] 動画の解像度が無効です (" << width << "x" << height << ")。ファイルが破損している可能性があります。" << std::endl << std::flush;
        return false;
    }

    // 2. アップスケーラーと補完器のロード
    int actual_tile_size = config.tile_size;
    if (actual_tile_size <= 0) {
        actual_tile_size = get_auto_tile_size();
        std::cout << "[INFO] タイルサイズを自動設定しました: " << actual_tile_size << std::endl << std::flush;
    }

    if (config.progress_callback) config.progress_callback(0.05f, "AIモデルのロード中...");
    std::cout << "[RUN] Step 3: AIモデルロード中..." << std::endl << std::flush;
    
    std::string actual_model_name = config.model_name;
    if (actual_model_name.empty()) {
        if (config.use_fast_model) {
            actual_model_name = "realesr-animevideov3-x" + std::to_string(config.scale);
        } else {
            actual_model_name = (config.scale == 2) ? "realesrgan-x2plus" : "realesrgan-x4plus";
        }
    }

    if (!upscaler.load(config.model_dir, config.scale, actual_tile_size, actual_model_name)) {
        std::cerr << "[ERROR] AIモデルのロードに失敗しました" << std::endl << std::flush;
        return false;
    }
    std::cout << "[RUN] Step 3: AIモデルロード完了" << std::endl << std::flush;
    
    if (config.enable_interpolation) {
        std::cout << "[RUN] Step 3b: RIFE モデルロード中..." << std::endl << std::flush;
        if (!interpolator_svc.load(config.model_dir)) {
            std::cerr << "[ERROR] RIFEモデルのロードに失敗しました" << std::endl << std::flush;
            return false;
        }
        std::cout << "[RUN] Step 3b: RIFE モデルロード完了" << std::endl << std::flush;
    }

    // 3. 出力設定
    std::string temp_video = "temp_no_audio.mp4";
    
    int start_frame = 0;
    int max_trim_frames = total_frames;

    if (config.enable_trim) {
        start_frame = static_cast<int>(config.trim_start_sec * fps);
        max_trim_frames = static_cast<int>(config.trim_duration_sec * fps);
        if (total_frames > 0 && start_frame >= total_frames) {
            std::cerr << "[ERROR] トリミング開始位置が動画の長さを超えています。/ Trim start position exceeds video duration." << std::endl;
            return false;
        }
        if (use_stabilizer) {
            for (int i = 0; i < start_frame; ++i) {
                stabilizer_svc.next_frame();
            }
        } else {
            cap.set(cv::CAP_PROP_POS_MSEC, config.trim_start_sec * 1000.0);
        }
        std::cout << "[INFO] トリミング有効: 開始フレーム=" << start_frame << ", 最大処理フレーム数=" << max_trim_frames << std::endl;
    }

    int total_to_process = config.enable_trim ? std::min(total_frames - start_frame, max_trim_frames) : total_frames;
    if (total_to_process <= 0) total_to_process = total_frames; // 安全策

    std::cout << "[INFO] 処理を開始します..." << std::endl;
    if (config.progress_callback) config.progress_callback(0.1f, "フレーム処理中...");

    auto start_time = std::chrono::steady_clock::now();

    // 制御変数とキュー
    std::atomic<bool> error_occurred(false);
    std::atomic<bool> done_decoding(false);
    SafeQueue<cv::Mat> decode_queue(5); // VRAM/RAM 節約のためバッファサイズは5
    SafeQueue<cv::Mat> encode_queue(5);

    // デコードスレッドの開始
    std::thread decoder_thread([&]() {
        cv::Mat local_frame;
        int decoded_count = 0;
        while (true) {
            if ((config.stop_flag && *config.stop_flag) || error_occurred) {
                break;
            }
            if (config.enable_trim && decoded_count >= max_trim_frames) {
                break;
            }

            if (use_stabilizer) {
                local_frame = stabilizer_svc.next_frame();
            } else {
                cap.read(local_frame);
            }

            if (local_frame.empty()) {
                break;
            }

            decode_queue.push(local_frame.clone()); // キューにプッシュ
            decoded_count++;
        }
        done_decoding = true;
        decode_queue.shutdown();
    });

    // エンコードスレッドの開始
    std::thread encoder_thread([&]() {
        cv::VideoWriter cv_writer;
        FILE* ffmpeg_pipe = nullptr;
        bool use_ffmpeg = false;
        bool writer_initialized = false;
        cv::Mat local_out;

        while (encode_queue.pop(local_out)) {
            if (error_occurred) {
                break;
            }

            if (!writer_initialized) {
                int target_width = local_out.cols;
                int target_height = local_out.rows;
                double output_fps = config.enable_interpolation ? fps * 2 : fps;

                // FFmpeg パイプ出力の試行
                if (utils::check_ffmpeg_available()) {
                    std::string ffmpeg_path = "ffmpeg";
#ifdef _WIN32
                    if (std::system("where ffmpeg > nul 2>&1") != 0) {
                        if (std::filesystem::exists("ffmpeg.exe")) {
                            ffmpeg_path = ".\\ffmpeg.exe";
                        }
                    }
#endif
                    // HW NVENC エンコーダ (4K超過時はHEVCにフォールバック)
                    std::string vcodec = (target_width > 4096 || target_height > 4096) ? "hevc_nvenc" : "h264_nvenc";
                    std::string cmd = ffmpeg_path + " -y -f rawvideo -pix_fmt bgr24 -s " + std::to_string(target_width) + "x" + std::to_string(target_height) +
                                      " -r " + std::to_string(output_fps) + " -i - -c:v " + vcodec + " -preset p4 -cq 20 -pix_fmt yuv420p \"" + temp_video + "\"";
#ifdef _WIN32
                    ffmpeg_pipe = _popen(cmd.c_str(), "wb");
#else
                    ffmpeg_pipe = popen(cmd.c_str(), "w");
#endif
                    if (ffmpeg_pipe) {
                        // 1フレーム書き込みテスト
                        size_t expected_size = local_out.total() * local_out.elemSize();
                        size_t written = fwrite(local_out.data, 1, expected_size, ffmpeg_pipe);
                        fflush(ffmpeg_pipe);
                        if (written == expected_size) {
                            use_ffmpeg = true;
                            std::cout << "[INFO] FFmpeg NVENC パイプエンコードを有効化しました。" << std::endl;
                        } else {
                            std::cerr << "[WARNING] FFmpeg パイプ(NVENC)の書き込みテストに失敗しました。CPUエンコードに切り替えます。" << std::endl;
#ifdef _WIN32
                            _pclose(ffmpeg_pipe);
#else
                            pclose(ffmpeg_pipe);
#endif
                            ffmpeg_pipe = nullptr;
                        }
                    }

                    if (!use_ffmpeg) {
                        // CPU libx264 で試行
                        std::string cmd_sw = ffmpeg_path + " -y -f rawvideo -pix_fmt bgr24 -s " + std::to_string(target_width) + "x" + std::to_string(target_height) +
                                          " -r " + std::to_string(output_fps) + " -i - -c:v libx264 -preset ultrafast -crf 20 -pix_fmt yuv420p \"" + temp_video + "\"";
#ifdef _WIN32
                        ffmpeg_pipe = _popen(cmd_sw.c_str(), "wb");
#else
                        ffmpeg_pipe = popen(cmd_sw.c_str(), "w");
#endif
                        if (ffmpeg_pipe) {
                            size_t expected_size = local_out.total() * local_out.elemSize();
                            size_t written = fwrite(local_out.data, 1, expected_size, ffmpeg_pipe);
                            fflush(ffmpeg_pipe);
                            if (written == expected_size) {
                                use_ffmpeg = true;
                                std::cout << "[INFO] FFmpeg libx264 パイプエンコードを有効化しました。" << std::endl;
                            } else {
                                std::cerr << "[WARNING] FFmpeg パイプ(libx264)の書き込みテストに失敗しました。" << std::endl;
#ifdef _WIN32
                                _pclose(ffmpeg_pipe);
#else
                                pclose(ffmpeg_pipe);
#endif
                                ffmpeg_pipe = nullptr;
                            }
                        }
                    }
                }

                if (!use_ffmpeg) {
                    std::cout << "[INFO] OpenCV VideoWriter を使用します。" << std::endl;
                    cv_writer.open(temp_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), output_fps, cv::Size(target_width, target_height));
                    if (!cv_writer.isOpened()) {
                        std::cerr << "[ERROR] 出力動画ファイルを開けません: " << temp_video << std::endl;
                        error_occurred = true;
                        break;
                    }
                    cv_writer.write(local_out);
                }
                writer_initialized = true;
            } else {
                if (use_ffmpeg && ffmpeg_pipe) {
                    size_t expected_size = local_out.total() * local_out.elemSize();
                    size_t written = fwrite(local_out.data, 1, expected_size, ffmpeg_pipe);
                    if (written < expected_size) {
                        std::cerr << "[ERROR] FFmpeg パイプ書き込み中にエラーが発生しました。" << std::endl;
                        error_occurred = true;
                        break;
                    }
                } else {
                    cv_writer.write(local_out);
                }
            }
        }

        if (use_ffmpeg && ffmpeg_pipe) {
#ifdef _WIN32
            _pclose(ffmpeg_pipe);
#else
            pclose(ffmpeg_pipe);
#endif
        } else {
            cv_writer.release();
        }
    });

    // メインスレッド: AI処理ループ
    cv::Mat frame, upscaled, prev_upscaled;
    cv::Mat prev_frame; // RIFE バイパス用
    int current_frame = 0;

    while (decode_queue.pop(frame)) {
        if ((config.stop_flag && *config.stop_flag) || error_occurred) {
            break;
        }

        // AI アップスケーリング
        if (!upscaler.process(frame, upscaled)) {
            std::cerr << "\n[ERROR] アップスケール失敗。" << std::endl;
            error_occurred = true;
            break;
        }

        // フレーム補間 (RIFE)
        if (config.enable_interpolation && !prev_upscaled.empty()) {
            bool is_static = false;
            // 静止フレーム判定 (元解像度のフレームで判定して高速化)
            if (!prev_frame.empty()) {
                cv::Mat diff;
                cv::absdiff(frame, prev_frame, diff);
                cv::Scalar mean_diff = cv::mean(diff);
                double diff_val = (mean_diff[0] + mean_diff[1] + mean_diff[2]) / 3.0;

                // しきい値 0.3
                if (diff_val < 0.3) {
                    is_static = true;
                }
            }

            cv::Mat interp_frame;
            if (is_static) {
                // RIFE推論をバイパスして、前フレームのコピーを使用
                interp_frame = prev_upscaled.clone();
            } else {
                if (!interpolator_svc.process(prev_upscaled, upscaled, interp_frame)) {
                    std::cerr << "\n[ERROR] RIFE処理失敗。" << std::endl;
                    error_occurred = true;
                    break;
                }
            }
            encode_queue.push(std::move(interp_frame));
        }

        // 本フレームをエンコードキューへプッシュ
        encode_queue.push(upscaled.clone());

        // RIFEバイパス用の前フレーム保持
        prev_frame = frame.clone();

        // 状態更新
        std::swap(prev_upscaled, upscaled);
        current_frame++;

        if (total_to_process > 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::max(std::chrono::duration<double>(now - start_time).count(), 0.001);
            double fps_processing = current_frame / elapsed_sec;
            double remaining_sec = (total_to_process - current_frame) / std::max(fps_processing, 0.001);
            
            float p = 0.1f + 0.8f * ((float)current_frame / total_to_process);
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1);
            ss << "処理中: " << current_frame << "/" << total_to_process << " フレーム"
               << " | " << fps_processing << " fps | 残り時間: ";
               
            int r_h = static_cast<int>(remaining_sec) / 3600;
            int r_m = (static_cast<int>(remaining_sec) % 3600) / 60;
            int r_s = static_cast<int>(remaining_sec) % 60;
            ss << std::setfill('0') << std::setw(2) << r_h << ":"
               << std::setw(2) << r_m << ":" << std::setw(2) << r_s;
               
            if (config.progress_callback) {
                config.progress_callback(p, ss.str());
            } else {
                std::cout << "\r" << ss.str() << std::flush;
            }
        } else {
            std::cout << "\rProcessed frames: " << current_frame << std::flush;
        }
    }

    // キューの終了とスレッド合流
    decode_queue.shutdown();
    encode_queue.shutdown();

    if (decoder_thread.joinable()) {
        decoder_thread.join();
    }
    if (encoder_thread.joinable()) {
        encoder_thread.join();
    }

    if (!use_stabilizer) cap.release();

    if (error_occurred) {
        if (fs::exists(temp_video)) {
            fs::remove(temp_video);
        }
        return false;
    }

    // 5. 音声マージ
    if (config.progress_callback) config.progress_callback(0.95f, "音声のマージ中...");
    bool merge_ok = utils::merge_audio(temp_video, config.input_path, config.output_path, config.enable_trim, config.trim_start_sec, config.trim_duration_sec);

    if (merge_ok && fs::exists(temp_video)) {
        fs::remove(temp_video);
    }

    // エラーパスでも一時ファイルを確実に削除
    if (!merge_ok && fs::exists(temp_video)) {
        fs::remove(temp_video);
    }

    if (config.progress_callback) config.progress_callback(1.0f, "完了しました！");
    return merge_ok;
}
