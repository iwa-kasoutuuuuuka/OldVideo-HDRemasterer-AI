#include "interpolator.h"
#include "rife_warp.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <filesystem>
#include "cpu.h"

Interpolator::Interpolator() : gpudevice(false), is_flownet(false), gpu_index(-1) {
    // CPU スレッド数の最適化
    rife.opt.num_threads = 1;
    
    // カスタムレイヤー "rife.Warp" を登録
    rife.register_custom_layer("rife.Warp", Warp_layer_creator);
}

Interpolator::~Interpolator() {
    rife.clear();
}

bool Interpolator::load(const std::string& model_dir, int gpu_index) {
    this->gpu_index = gpu_index;
    if (gpu_index >= 0) {
        gpudevice = true;
        rife.opt.use_vulkan_compute = true;
        rife.opt.use_fp16_packed = true;
        rife.opt.use_fp16_storage = true;
        rife.opt.use_fp16_arithmetic = true;
        rife.opt.use_shader_pack8 = true;
        rife.opt.use_image_storage = false; // Vulkan最大次元(16384等)エラー回避のため
        rife.set_vulkan_device(gpu_index);
    } else {
        gpudevice = false;
        rife.opt.use_vulkan_compute = false;
    }

    std::string param = model_dir + "/rife-v4.param";
    std::string bin = model_dir + "/rife-v4.bin";
    is_flownet = false;

    if (!std::filesystem::exists(param)) {
        if (std::filesystem::exists(model_dir + "/flownet.param")) {
            param = model_dir + "/flownet.param";
            bin = model_dir + "/flownet.bin";
            is_flownet = true;
        }
    }
    
    model_param = param;
    model_bin = bin;

    if (rife.load_param(param.c_str()) != 0) {
        std::cerr << "[ERROR] RIFE param load failed: " << param << std::endl;
        return false;
    }
    if (rife.load_model(bin.c_str()) != 0) {
        std::cerr << "[ERROR] RIFE model load failed: " << bin << std::endl;
        return false;
    }

    std::cout << "[INFO] RIFE モデルロード完了 (動作モード: " << (gpudevice ? "GPU/Vulkan" : "CPU") << ", GPU Index: " << gpu_index << ")" << std::endl << std::flush;
    return true;
}

bool Interpolator::switch_to_cpu() {
    std::cout << "[WARNING] RIFE GPU 処理に失敗しました。CPU モードに切り替えます... / RIFE GPU processing failed. Switching to CPU mode..." << std::endl;
    gpudevice = false;
    rife.clear();
    rife.opt.use_vulkan_compute = false;
    
    if (rife.load_param(model_param.c_str()) != 0) return false;
    if (rife.load_model(model_bin.c_str()) != 0) return false;
    return true;
}

bool Interpolator::process(const cv::Mat& f1, const cv::Mat& f2, cv::Mat& out, float timestep) {
    if (f1.empty() || f2.empty()) return false;

    int w = f1.cols;
    int h = f1.rows;
    out.create(h, w, CV_8UC3);

    // 入出力名の決定
    std::string in0_name = is_flownet ? "in0" : "input0";
    std::string in1_name = is_flownet ? "in1" : "input1";
    std::string in2_name = is_flownet ? "in2" : "";
    std::string out_name = is_flownet ? "out0" : "output";

    ncnn::Mat n_timestep(1);
    n_timestep[0] = timestep;

    int tile_size = 2048;

    if (w <= tile_size && h <= tile_size) {
        int ret = 0;
        {
            ncnn::Mat n_f1 = ncnn::Mat::from_pixels(f1.data, ncnn::Mat::PIXEL_BGR, w, h);
            ncnn::Mat n_f2 = ncnn::Mat::from_pixels(f2.data, ncnn::Mat::PIXEL_BGR, w, h);

            std::cout << "[DEBUG] RIFE 一括推論開始: size=" << w << "x" << h << std::endl << std::flush;
            ncnn::Extractor ex = rife.create_extractor();
            ex.input(in0_name.c_str(), n_f1);
            ex.input(in1_name.c_str(), n_f2);
            if (!in2_name.empty()) ex.input(in2_name.c_str(), n_timestep);
            
            ncnn::Mat n_out;
            ret = ex.extract(out_name.c_str(), n_out);
            std::cout << "[DEBUG] RIFE 一括推論終了 (ret=" << ret << ")" << std::endl << std::flush;
            if (ret == 0) {
                n_out.to_pixels(out.data, ncnn::Mat::PIXEL_BGR);
            }
        }
        
        if (ret != 0) {
            if (gpudevice && switch_to_cpu()) return process(f1, f2, out, timestep);
            return false;
        }
        return true;
    }

    // Tiled processing for RIFE
    const int padding = 32;
    int tile_count = 0;
    for (int y = 0; y < h; y += tile_size) {
        for (int x = 0; x < w; x += tile_size) {
            int x0 = std::max(x - padding, 0);
            int y0 = std::max(y - padding, 0);
            int x1 = std::min(x + tile_size + padding, w);
            int y1 = std::min(y + tile_size + padding, h);

            cv::Mat tile_f1, tile_f2, tile_out;
            f1(cv::Range(y0, y1), cv::Range(x0, x1)).copyTo(tile_f1);
            f2(cv::Range(y0, y1), cv::Range(x0, x1)).copyTo(tile_f2);
            int ret = 0;
            {
                ncnn::Mat nt1 = ncnn::Mat::from_pixels(tile_f1.data, ncnn::Mat::PIXEL_BGR, tile_f1.cols, tile_f1.rows);
                ncnn::Mat nt2 = ncnn::Mat::from_pixels(tile_f2.data, ncnn::Mat::PIXEL_BGR, tile_f2.cols, tile_f2.rows);

                std::cout << "[DEBUG] RIFE tile " << tile_count << " (x=" << x << ", y=" << y << ") 推論開始..." << std::endl << std::flush;
                ncnn::Extractor ex = rife.create_extractor();
                ex.input(in0_name.c_str(), nt1);
                ex.input(in1_name.c_str(), nt2);
                if (!in2_name.empty()) ex.input(in2_name.c_str(), n_timestep);
                
                ncnn::Mat nt_out;
                ret = ex.extract(out_name.c_str(), nt_out);
                std::cout << "[DEBUG] RIFE tile " << tile_count << " 推論終了 (ret=" << ret << ")" << std::endl << std::flush;

                if (ret == 0) {
                    tile_out.create(nt_out.h, nt_out.w, CV_8UC3);
                    nt_out.to_pixels(tile_out.data, ncnn::Mat::PIXEL_BGR);
                }
            }

            if (ret != 0) {
                if (gpudevice && switch_to_cpu()) return process(f1, f2, out, timestep);
                return false;
            }

            int target_w = std::min(tile_size, w - x);
            int target_h = std::min(tile_size, h - y);
            int src_x = x - x0;
            int src_y = y - y0;

            // tile_out のサイズに合わせた安全な境界クリップを追加
            if (src_x + target_w > tile_out.cols) {
                target_w = std::max(tile_out.cols - src_x, 0);
            }
            if (src_y + target_h > tile_out.rows) {
                target_h = std::max(tile_out.rows - src_y, 0);
            }

            if (target_w > 0 && target_h > 0) {
                tile_out(cv::Range(src_y, src_y + target_h), cv::Range(src_x, src_x + target_w))
                    .copyTo(out(cv::Range(y, y + target_h), cv::Range(x, x + target_w)));
            }
            tile_count++;
        }
    }

    return true;
}
