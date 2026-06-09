// -*- coding: utf-8 -*-
#include "upscaler.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include "cpu.h"

Upscaler::Upscaler() : scale(4), tile_size(0), gpudevice(false), gpu_index(-1) {
    // GPU 自動選択
    gpu_index = utils::get_preferred_gpu_index();
    if (gpu_index >= 0) {
        gpudevice = true;
        // ncnn 高速化オプション
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
        net.opt.use_int8_storage = true;
        net.opt.use_shader_pack8 = true;
        net.opt.use_image_storage = false; // Vulkan最大次元(16384等)エラー回避のためfalse
    }
    // CPU スレッド数の最適化
    net.opt.num_threads = ncnn::get_cpu_count();
}

Upscaler::~Upscaler() {
    net.clear();
}

bool Upscaler::load(const std::string& model_dir, int _scale, int _tile_size, const std::string& _model_name) {
    this->scale = _scale;
    this->tile_size = _tile_size;

    std::string model_name = _model_name;
    if (model_name.empty()) {
        // デフォルト: アニメモデルをフォールバックとして使用
        model_name = "realesr-animevideov3-x" + std::to_string(scale);
    }

    model_param = model_dir + "/" + model_name + ".param";
    model_bin = model_dir + "/" + model_name + ".bin";
    
    std::cout << "[INFO] AIモデル: " << model_name << std::endl;

    if (gpudevice) {
        net.set_vulkan_device(gpu_index);
    }

    if (net.load_param(model_param.c_str()) != 0) {
        std::cerr << "[ERROR] Failed to load model param: " << model_param << std::endl;
        return false;
    }
    if (net.load_model(model_bin.c_str()) != 0) {
        std::cerr << "[ERROR] Failed to load model weights: " << model_bin << std::endl;
        return false;
    }

    return true;
}

bool Upscaler::switch_to_cpu() {
    std::cout << "[WARNING] GPU 処理に失敗しました。VRAM 不足の可能性があります。CPU モードに切り替えます... / GPU processing failed. Switching to CPU mode..." << std::endl;
    gpudevice = false;
    net.clear();
    net.opt.use_vulkan_compute = false;
    
    if (net.load_param(model_param.c_str()) != 0) return false;
    if (net.load_model(model_bin.c_str()) != 0) return false;
    return true;
}

bool Upscaler::process(const cv::Mat& in, cv::Mat& out) {
    if (in.empty()) return false;

    int w = in.cols;
    int h = in.rows;

    int out_w = w * scale;
    int out_h = h * scale;
    out.create(out_h, out_w, CV_8UC3);
    if (out.empty()) {
        std::cerr << "[ERROR] 出力バッファのメモリ確保に失敗しました (" << out_w << "x" << out_h << ")" << std::endl;
        return false;
    }

    // Full image processing
    if (tile_size <= 0 || (tile_size >= w && tile_size >= h)) {
        int ret = 0;
        {
            ncnn::Mat n_in = ncnn::Mat::from_pixels(in.data, ncnn::Mat::PIXEL_BGR, w, h);
            ncnn::Extractor ex = net.create_extractor();
            ex.input("data", n_in);
            ncnn::Mat n_out;
            ret = ex.extract("output", n_out);
            if (ret == 0) {
                // ncnn出力サイズの検証（バッファオーバーフロー防止）
                if (n_out.w != out_w || n_out.h != out_h) {
                    std::cerr << "[ERROR] ncnn出力サイズ不一致: 期待=" << out_w << "x" << out_h << ", 実際=" << n_out.w << "x" << n_out.h << std::endl;
                    ret = -1;
                } else {
                    n_out.to_pixels(out.data, ncnn::Mat::PIXEL_BGR);
                }
            }
        }
        
        if (ret != 0) {
            if (gpudevice && switch_to_cpu()) return process(in, out);
            return false;
        }
        return true;
    }

    // Tiled processing for VRAM conservation
    const int padding = 32;
    int tile_w = tile_size;
    int tile_h = tile_size;

    for (int y = 0; y < h; y += tile_h) {
        for (int x = 0; x < w; x += tile_w) {
            int x0 = std::max(x - padding, 0);
            int y0 = std::max(y - padding, 0);
            int x1 = std::min(x + tile_w + padding, w);
            int y1 = std::min(y + tile_h + padding, h);

            in(cv::Range(y0, y1), cv::Range(x0, x1)).copyTo(tile_in);
            int ret = 0;
            {
                ncnn::Mat n_tile_in = ncnn::Mat::from_pixels(tile_in.data, ncnn::Mat::PIXEL_BGR, tile_in.cols, tile_in.rows);
                ncnn::Extractor ex = net.create_extractor();
                ex.input("data", n_tile_in);
                ncnn::Mat n_tile_out;
                ret = ex.extract("output", n_tile_out);

                if (ret == 0) {
                    tile_out.create(n_tile_out.h, n_tile_out.w, CV_8UC3);
                    n_tile_out.to_pixels(tile_out.data, ncnn::Mat::PIXEL_BGR);
                }
            }

            if (ret != 0) {
                if (gpudevice && switch_to_cpu()) return process(in, out);
                return false;
            }

            int target_x = x * scale;
            int target_y = y * scale;
            int target_w_tile = std::min(tile_w, w - x) * scale;
            int target_h_tile = std::min(tile_h, h - y) * scale;

            int src_x = (x - x0) * scale;
            int src_y = (y - y0) * scale;

            tile_out(cv::Range(src_y, src_y + target_h_tile), cv::Range(src_x, src_x + target_w_tile))
                .copyTo(out(cv::Range(target_y, target_y + target_h_tile), cv::Range(target_x, target_x + target_w_tile)));
        }
    }

    return true;
}
