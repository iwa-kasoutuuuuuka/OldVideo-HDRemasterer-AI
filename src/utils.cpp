#include "utils.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "gpu.h"

namespace utils {

    int get_preferred_gpu_index() {
        int count = ncnn::get_gpu_count();
        if (count == 0) return -1;
        if (count == 1) return 0;
        
        // 複数 GPU がある場合、最後のデバイス（通常ディスクリート GPU）を優先
        // Intel 内蔵 GPU はインデックス 0 に列挙されることが多い
        int preferred = count - 1;
        std::cout << "[INFO] GPU " << count << " 台検出。デバイス " << preferred << " を使用します。" << std::endl;
        return preferred;
    }

    void denoise_frame(const cv::Mat& input, cv::Mat& output) {
        // fastNlMeansDenoisingColored は重いためスキップし、ノイズ処理は AI (Real-ESRGAN 等) に任せる
        input.copyTo(output);
    }

    void enhance_frame(const cv::Mat& input, cv::Mat& output) {
        // 重い CLAHE やガウシアンブラーを省略し、軽量なコントラスト微調整のみ適用して高速化
        input.convertTo(output, -1, 1.05, 0); // alpha=1.05, beta=0 でわずかにコントラストを上げる
    }

    bool merge_audio(const std::string& video_no_audio, const std::string& original_video, const std::string& final_output) {
        std::cout << "[INFO] 音声のマージを開始します..." << std::endl;
        
        // FFmpeg の存在確認 (PATH 内の ffmpeg を使用)
        std::string ffmpeg_cmd = "ffmpeg";
        
        // システムの ffmpeg が使えるか確認 (デバッグ用)
        #ifdef _WIN32
        if (std::system("where ffmpeg > nul 2>&1") != 0) {
            std::cerr << "[WARNING] PATH に ffmpeg が見つかりません。デフォルトの場所や PATH 設定を確認してください。" << std::endl;
            // 必要に応じて手動パス指定を検討
        }
        #else
        if (std::system("which ffmpeg > /dev/null 2>&1") != 0) {
            std::cerr << "[WARNING] PATH に ffmpeg が見つかりません。" << std::endl;
        }
        #endif

        // FFmpegコマンド: NVENC ハードウェアエンコードを試行 (-c:v h264_nvenc)
        std::string cmd_hw = ffmpeg_cmd + " -y -i \"" + video_no_audio + "\" -i \"" + original_video + 
                          "\" -c:v h264_nvenc -preset p6 -cq 20 -c:a aac -map 0:v:0 -map 1:a:0? -shortest \"" + final_output + "\"";
        
        int ret = std::system(cmd_hw.c_str());
        if (ret != 0) {
            std::cout << "[INFO] Hardware encoding (NVENC) failed or unavailable. Falling back to libx264 software encoding." << std::endl;
            // 失敗した場合は libx264 ソフトウェアエンコードにフォールバック
            std::string cmd_sw = ffmpeg_cmd + " -y -i \"" + video_no_audio + "\" -i \"" + original_video + 
                              "\" -c:v libx264 -crf 20 -preset fast -c:a aac -map 0:v:0 -map 1:a:0? -shortest \"" + final_output + "\"";
            ret = std::system(cmd_sw.c_str());
        }
        return (ret == 0);
    }

    void print_progress(int current, int total) {
        float progress = (float)current / total * 100.0f;
        std::cout << "\r[PROGRESS] " << std::fixed << std::setprecision(2) << progress << "% (" 
                  << current << " / " << total << " )" << std::flush;
        if (current == total) std::cout << std::endl;
    }

}
