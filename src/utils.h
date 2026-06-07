#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

/**
 * @brief 画像処理・GPU選択ユーティリティ
 */
namespace utils {

    /**
     * @brief 推奨 GPU デバイスインデックスを返す
     * NVIDIA/AMD などのディスクリート GPU を優先し、
     * Intel 内蔵 GPU より高速なデバイスを自動選択する。
     * GPU が見つからない場合は -1 を返す。
     */
    int get_preferred_gpu_index();

    /**
     * @brief フレームのノイズ除去
     */
    void denoise_frame(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief フレームの画質調整
     */
    void enhance_frame(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief パス文字列のサニタイズ（コマンドインジェクション対策）
     */
    std::string sanitize_path(const std::string& path);

    /**
     * @brief FFmpeg の存在を確認
     */
    bool check_ffmpeg_available();

    /**
     * @brief FFmpeg を使用して音声付き動画を作成 (トリミング対応)
     */
    bool merge_audio(const std::string& video_no_audio, const std::string& original_video, const std::string& final_output, bool enable_trim = false, double trim_start = 0.0, double trim_duration = 0.0);

    /**
     * @brief 進捗表示
     */
    void print_progress(int current, int total);

}

#endif // UTILS_H
