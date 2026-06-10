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

int VideoProcessor::get_auto_tile_size(bool enable_interpolation, const std::string& model_name) {
    int gpu_idx = utils::get_preferred_gpu_index();
    if (gpu_idx < 0) return 256; // Fallback for CPU
    uint32_t budget_mb = ncnn::get_gpu_device(gpu_idx)->get_heap_budget();
    
    // realesrgan-x4plus や realesrgan-x2plus などの重い実写モデルか判定
    bool is_heavy_model = (model_name.find("realesrgan-x") != std::string::npos);
    
    // RIFE補間が有効な場合、または重い実写モデルの場合、VRAM溢れを防ぐためにタイルサイズを制限する
    if (enable_interpolation || is_heavy_model) {
        if (budget_mb >= 6000) return 320;
        if (budget_mb >= 3000) return 256; // GTX 1650等 (4GB VRAM) では 256 が安全
        if (budget_mb >= 1500) return 128;
        return 96;
    } else {
        if (budget_mb >= 6000) return 400;
        if (budget_mb >= 3000) return 320;
        if (budget_mb >= 1500) return 160;
        return 128;
    }
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
    std::string actual_model_name = config.model_name;
    if (actual_model_name.empty()) {
        if (config.use_fast_model) {
            actual_model_name = "realesr-animevideov3-x" + std::to_string(config.scale);
        } else {
            actual_model_name = (config.scale == 2) ? "realesrgan-x2plus" : "realesrgan-x4plus";
        }
    }

    if (actual_tile_size <= 0) {
        actual_tile_size = get_auto_tile_size(config.enable_interpolation, actual_model_name);
        std::cout << "[INFO] タイルサイズを自動設定しました: " << actual_tile_size << std::endl << std::flush;
    }

    if (config.progress_callback) config.progress_callback(0.05f, "AIモデルのロード中...");
    std::cout << "[RUN] Step 3: AIモデルロード中..." << std::endl << std::flush;
    
    if (!upscaler.load(config.model_dir, config.scale, actual_tile_size, actual_model_name)) {
        std::cerr << "[ERROR] AIモデルのロードに失敗しました" << std::endl << std::flush;
        return false;
    }
    std::cout << "[RUN] Step 3: AIモデルロード完了" << std::endl << std::flush;
    
    if (config.enable_interpolation) {
        std::cout << "[RUN] Step 3b: RIFE モデルロード中..." << std::endl << std::flush;
        int gpu_idx = utils::get_preferred_gpu_index();
        
        // マルチGPU検出と自動割り当て
        std::vector<int> gpu_list = utils::get_gpu_indices_sorted();
        if (gpu_list.size() > 1) {
            gpu_idx = gpu_list[1]; // セカンドGPUを使用
            std::cout << "[INFO] マルチGPU検出: RIFE補間にセカンドGPU (Index: " << gpu_idx << ") を割り当てます。" << std::endl << std::flush;
        } else {
            std::cout << "[INFO] シングルGPU検出: RIFE補間と超解像で同一GPU (Index: " << gpu_idx << ") を共有します。" << std::endl << std::flush;
        }

        if (!interpolator_svc.load(config.model_dir, gpu_idx)) {
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
    SafeQueue<cv::Mat> upscale_queue(16);
    SafeQueue<cv::Mat> encode_queue(16);

    // デコード＆補間スレッドの開始 (デコードと低解像度でのRIFE補間を直列化してスレッド競合を防ぐ)
    std::thread decoder_thread([&]() {
        cv::Mat local_frame;
        cv::Mat prev_frame;
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

            // フレーム補間 (RIFE)
            if (config.enable_interpolation && !prev_frame.empty()) {
                bool is_static = false;
                // 静止フレーム判定
                cv::Mat diff;
                cv::absdiff(local_frame, prev_frame, diff);
                cv::Scalar mean_diff = cv::mean(diff);
                double diff_val = (mean_diff[0] + mean_diff[1] + mean_diff[2]) / 3.0;

                // しきい値 0.3
                if (diff_val < 0.3) {
                    is_static = true;
                }

                cv::Mat interp_frame;
                if (is_static) {
                    // RIFE推論をバイパスして、前フレームのコピーを使用
                    interp_frame = prev_frame.clone();
                } else {
                    std::cout << "[DEBUG] RIFE 推論開始: size=" << prev_frame.cols << "x" << prev_frame.rows << std::endl << std::flush;
                    if (!interpolator_svc.process(prev_frame, local_frame, interp_frame)) {
                        std::cerr << "\n[ERROR] RIFE処理失敗。" << std::endl;
                        error_occurred = true;
                        break;
                    }
                    std::cout << "[DEBUG] RIFE 推論成功" << std::endl << std::flush;
                }
                upscale_queue.push(std::move(interp_frame));
            }

            // 本フレームをアップスケールキューへプッシュ
            if (config.enable_interpolation) {
                prev_frame = local_frame.clone(); // 次回補間用にクローンして保持
                upscale_queue.push(std::move(local_frame)); // 本フレームは move で転送
            } else {
                upscale_queue.push(std::move(local_frame)); // 補間なしならクローンなしで move
            }
            decoded_count++;
        }
        upscale_queue.shutdown();
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
                    // 入力ピクセルフォーマットを yuv420p に変更してデータ転送量を半減＆FFmpeg swscaleを回避
                    std::string cmd = ffmpeg_path + " -y -f rawvideo -pix_fmt yuv420p -s " + std::to_string(target_width) + "x" + std::to_string(target_height) +
                                      " -r " + std::to_string(output_fps) + " -i - -c:v " + vcodec + " -preset p2 -cq 20 -pix_fmt yuv420p \"" + temp_video + "\" 2> ffmpeg_error.log";
#ifdef _WIN32
                    ffmpeg_pipe = _popen(cmd.c_str(), "wb");
#else
                    ffmpeg_pipe = popen(cmd.c_str(), "w");
#endif
                    if (ffmpeg_pipe) {
                        // 1フレーム書き込みテスト (YUV420pへの変換を行いサイズを確認)
                        cv::Mat yuv_test;
                        cv::cvtColor(local_out, yuv_test, cv::COLOR_BGR2YUV_I420);
                        size_t expected_size = yuv_test.total() * yuv_test.elemSize();
                        size_t written = fwrite(yuv_test.data, 1, expected_size, ffmpeg_pipe);
                        fflush(ffmpeg_pipe);
                        if (written == expected_size) {
                            use_ffmpeg = true;
                            std::cout << "[INFO] FFmpeg NVENC パイプエンコードを有効化しました (YUV420p直接転送)。" << std::endl;
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
                        std::string cmd_sw = ffmpeg_path + " -y -f rawvideo -pix_fmt yuv420p -s " + std::to_string(target_width) + "x" + std::to_string(target_height) +
                                          " -r " + std::to_string(output_fps) + " -i - -c:v libx264 -preset ultrafast -crf 20 -pix_fmt yuv420p \"" + temp_video + "\" 2> ffmpeg_error.log";
#ifdef _WIN32
                        ffmpeg_pipe = _popen(cmd_sw.c_str(), "wb");
#else
                        ffmpeg_pipe = popen(cmd_sw.c_str(), "w");
#endif
                        if (ffmpeg_pipe) {
                            cv::Mat yuv_test;
                            cv::cvtColor(local_out, yuv_test, cv::COLOR_BGR2YUV_I420);
                            size_t expected_size = yuv_test.total() * yuv_test.elemSize();
                            size_t written = fwrite(yuv_test.data, 1, expected_size, ffmpeg_pipe);
                            fflush(ffmpeg_pipe);
                            if (written == expected_size) {
                                use_ffmpeg = true;
                                std::cout << "[INFO] FFmpeg libx264 パイプエンコードを有効化しました (YUV420p直接転送)。" << std::endl;
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
                    cv::Mat yuv_frame;
                    cv::cvtColor(local_out, yuv_frame, cv::COLOR_BGR2YUV_I420);
                    size_t expected_size = yuv_frame.total() * yuv_frame.elemSize();
                    std::cout << "[DEBUG] FFmpeg パイプ書き込み開始 (YUV): size=" << expected_size << std::endl << std::flush;
                    size_t written = fwrite(yuv_frame.data, 1, expected_size, ffmpeg_pipe);
                    if (written < expected_size) {
                        std::cerr << "[ERROR] FFmpeg パイプ書き込み中にエラーが発生しました。" << std::endl;
                        error_occurred = true;
                        break;
                    }
                    std::cout << "[DEBUG] FFmpeg パイプ書き込み成功" << std::endl << std::flush;
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

    // メインスレッド: GPU超解像専用処理ループ
    cv::Mat frame;
    int current_frame = 0;
    
    // 補間（RIFE）によって最終出力フレーム数は約2倍（補間有効時）になる
    // 進捗表示のために処理総フレーム数を調整
    int expected_total_to_process = config.enable_interpolation ? (total_to_process * 2 - 1) : total_to_process;
    if (expected_total_to_process <= 0) expected_total_to_process = 1; // 0除算防止

    while (upscale_queue.pop(frame)) {
        if ((config.stop_flag && *config.stop_flag) || error_occurred) {
            break;
        }

        cv::Mat upscaled;
        // AI アップスケーリング (GPU)
        std::cout << "[DEBUG] 超解像推論開始: frame=" << current_frame << ", size=" << frame.cols << "x" << frame.rows << std::endl << std::flush;
        if (!upscaler.process(frame, upscaled)) {
            std::cerr << "\n[ERROR] アップスケール失敗。" << std::endl;
            error_occurred = true;
            break;
        }
        std::cout << "[DEBUG] 超解像推論成功" << std::endl << std::flush;

        // エンコードスレッドへ引き渡す (std::moveでクローンを回避)
        encode_queue.push(std::move(upscaled));

        current_frame++;

        if (expected_total_to_process > 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::max(std::chrono::duration<double>(now - start_time).count(), 0.001);
            double fps_processing = current_frame / elapsed_sec;
            double remaining_sec = (expected_total_to_process - current_frame) / std::max(fps_processing, 0.001);
            
            float p = 0.1f + 0.8f * ((float)current_frame / expected_total_to_process);
            p = std::min(std::max(p, 0.0f), 0.9f); // 90% を進捗上限にして残りは音声マージなどの余白とする
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1);
            ss << "処理中: " << current_frame << "/" << expected_total_to_process << " フレーム"
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
    upscale_queue.shutdown();
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

