// -*- coding: utf-8 -*-
#ifndef UPSCALER_H
#define UPSCALER_H

#include <opencv2/opencv.hpp>
#include "net.h"
#include "gpu.h"

/**
 * @brief Real-ESRGAN ncnn wrapper class
 */
class Upscaler {
public:
    Upscaler();
    ~Upscaler();

    /**
     * @brief Load the model
     * @param model_dir Directory containing model files
     * @param scale Scale factor (default 4)
     * @param tile_size Tile split size (0 for auto)
     * @return Success or failure
     */
    bool load(const std::string& model_dir, int scale = 4, int tile_size = 0, bool use_fast_model = false);

    /**
     * @brief Upscale a frame
     * @param in Input frame (cv::Mat BGR)
     * @param out Output frame
     * @return Success or failure
     */
    bool process(const cv::Mat& in, cv::Mat& out);

private:
    ncnn::Net net;
    int scale;
    int tile_size;
    bool gpudevice;
    int gpu_index;
    
    bool switch_to_cpu();
    
    std::string model_param;
    std::string model_bin;
};

#endif // UPSCALER_H
