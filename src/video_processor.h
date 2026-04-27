#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H

#include <string>
#include <functional>
#include "upscaler.h"

/**
 * @brief 動画処理パイプラインの管理
 */
class VideoProcessor {
public:
    /**
     * @brief 処理設定
     */
    struct Config {
        std::string input_path;
        std::string output_path;
        int scale = 4;
        int tile_size = 0;
        std::string model_dir = "models";
        bool enable_stabilization = false;
        bool enable_interpolation = false;
        bool use_fast_model = false;  // true: 軽量モデル (animevideov3, 高速), false: 高品質モデル (x4plus)
        int num_gpus = 1;

        // 進捗通知用コールバック: float progress (0.0-1.0), string message
        std::function<void(float, std::string)> progress_callback = nullptr;
    };

    VideoProcessor();
    
    /**
     * @brief リマスター処理の実行
     * @param config 設定
     * @return 成功したか
     */
    bool run(const Config& config);

    /**
     * @brief VRAMから自動で推奨タイルサイズを計算します
     */
    static int get_auto_tile_size();

private:
    Upscaler upscaler;
    
    // 追加機能メンバー
    struct StabilizerContext; // Pimpl または forward decl
    struct InterpolatorContext;
};

#endif // VIDEO_PROCESSOR_H
