// -*- coding: utf-8 -*-
#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include <opencv2/opencv.hpp>
#include "net.h"
#include "gpu.h"

/**
 * @brief RIFE ncnn frame interpolation class
 */
class Interpolator {
public:
    Interpolator();
    ~Interpolator();

    /**
     * @brief Load the RIFE model
     * @param model_dir Directory containing rife model files
     * @return Success or failure
     */
    bool load(const std::string& model_dir);

    /**
     * @brief Generate an intermediate frame between two frames
     * @param f1 Previous frame
     * @param f2 Next frame
     * @param out Interpolated frame
     * @param timestep Interpolation position (0.5 = midpoint)
     * @return Success or failure
     */
    bool process(const cv::Mat& f1, const cv::Mat& f2, cv::Mat& out, float timestep = 0.5f);

private:
    ncnn::Net rife;
    bool gpudevice;
    bool is_flownet;
    int gpu_index;
};

#endif // INTERPOLATOR_H
