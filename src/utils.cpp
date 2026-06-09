#include "utils.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include "net.h"
#include "gpu.h"
#include <string>
#include <algorithm>

namespace utils {

    int get_preferred_gpu_index() {
        int count = ncnn::get_gpu_count();
        if (count == 0) return -1;
        if (count == 1) return 0;
        
        int best_idx = 0;
        int best_score = -1;
        
        for (int i = 0; i < count; i++) {
            const auto* device = ncnn::get_gpu_device(i);
            if (!device) continue;
            
            const auto& info = device->info;
            std::string name = info.device_name();
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            
            int score = 0;
            // デバイスタイプ判定 (Discrete GPU優先: type 0)
            if (info.type() == 0) {
                score += 1000;
            } else if (info.type() == 1) {
                score += 100;
            }
            
            // ベンダー名/ブランド名キーワード判定
            if (name.find("nvidia") != std::string::npos || name.find("geforce") != std::string::npos) {
                score += 500;
            } else if (name.find("amd") != std::string::npos || name.find("radeon") != std::string::npos) {
                score += 400;
            } else if (name.find("intel") != std::string::npos) {
                score += 50;
            }
            
            // VRAM予算（補助スコア）
            uint32_t budget = device->get_heap_budget();
            score += static_cast<int>(budget / 100);
            
            std::cout << "[INFO] GPU デバイス [" << i << "]: " << info.device_name() 
                      << " (Type: " << info.type() << ", VRAM: " << budget << " MB) -> スコア: " << score << std::endl;
            
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }
        
        const auto* best_device = ncnn::get_gpu_device(best_idx);
        std::cout << "[INFO] GPU 自動選択: " << best_idx << " [" << best_device->info.device_name() << "] を使用します。" << std::endl;
        return best_idx;
    }

    void denoise_frame(const cv::Mat& input, cv::Mat& output) {
        // fastNlMeansDenoisingColored は重いためスキップし、ノイズ処理は AI (Real-ESRGAN 等) に任せる
        input.copyTo(output);
    }

    void enhance_frame(const cv::Mat& input, cv::Mat& output) {
        // 重い CLAHE やガウシアンブラーを省略し、軽量なコントラスト微調整のみ適用して高速化
        input.convertTo(output, -1, 1.05, 0); // alpha=1.05, beta=0 でわずかにコントラストを上げる
    }

    std::string sanitize_path(const std::string& path) {
        std::string s = path;
        // OSコマンドインジェクションを防ぐため、シェルの特殊文字および改行コードを削除
        const std::string unsafe_chars = "\"\';&|`$<>^%\n\r";
        s.erase(std::remove_if(s.begin(), s.end(), [&unsafe_chars](char c) {
            return unsafe_chars.find(c) != std::string::npos;
        }), s.end());
        return s;
    }

    bool check_ffmpeg_available() {
        #ifdef _WIN32
        if (std::system("where ffmpeg > nul 2>&1") == 0) {
            return true;
        }
        if (std::filesystem::exists("ffmpeg.exe")) {
            return true;
        }
        #else
        if (std::system("which ffmpeg > /dev/null 2>&1") == 0) {
            return true;
        }
        if (std::filesystem::exists("./ffmpeg")) {
            return true;
        }
        #endif
        return false;
    }

    bool merge_audio(const std::string& video_no_audio, const std::string& original_video, const std::string& final_output, bool enable_trim, double trim_start, double trim_duration) {
        std::cout << "[INFO] 音声のマージを開始します..." << std::endl;
        
        std::string s_no_audio = sanitize_path(video_no_audio);
        std::string s_original = sanitize_path(original_video);
        std::string s_final = sanitize_path(final_output);
        
        std::string ffmpeg_cmd = "ffmpeg";
        #ifdef _WIN32
        if (std::system("where ffmpeg > nul 2>&1") != 0) {
            if (std::filesystem::exists("ffmpeg.exe")) {
                ffmpeg_cmd = ".\\ffmpeg.exe";
            } else {
                std::cerr << "[WARNING] ffmpeg がシステム PATH にもカレントディレクトリにも見つかりません。" << std::endl;
            }
        }
        #else
        if (std::system("which ffmpeg > /dev/null 2>&1") != 0) {
            if (std::filesystem::exists("./ffmpeg")) {
                ffmpeg_cmd = "./ffmpeg";
            } else {
                std::cerr << "[WARNING] ffmpeg が見つかりません。" << std::endl;
            }
        }
        #endif

        std::string trim_input_args = "";
        if (enable_trim) {
            std::stringstream ss;
            ss << " -ss " << trim_start << " -t " << trim_duration;
            trim_input_args = ss.str();
        }

        // FFmpegコマンド: NVENC ハードウェアエンコードを試行 (サニタイズ済みのパスを使用)
        std::string cmd_hw = ffmpeg_cmd + " -y -i \"" + s_no_audio + "\"" + trim_input_args + " -i \"" + s_original + 
                          "\" -c:v h264_nvenc -preset p6 -cq 20 -c:a aac -map 0:v:0 -map 1:a:0? -shortest \"" + s_final + "\"";
        
        int ret = std::system(cmd_hw.c_str());
        if (ret != 0) {
            std::cout << "[INFO] Hardware encoding (NVENC) failed or unavailable. Falling back to libx264 software encoding." << std::endl;
            std::string cmd_sw = ffmpeg_cmd + " -y -i \"" + s_no_audio + "\"" + trim_input_args + " -i \"" + s_original + 
                              "\" -c:v libx264 -crf 20 -preset fast -c:a aac -map 0:v:0 -map 1:a:0? -shortest \"" + s_final + "\"";
            ret = std::system(cmd_sw.c_str());
        }
        return (ret == 0);
    }

    void print_progress(int current, int total) {
        if (total <= 0) return;
        float progress = (float)current / total * 100.0f;
        std::cout << "\r[PROGRESS] " << std::fixed << std::setprecision(2) << progress << "% (" 
                  << current << " / " << total << " )" << std::flush;
        if (current == total) std::cout << std::endl;
    }

}
