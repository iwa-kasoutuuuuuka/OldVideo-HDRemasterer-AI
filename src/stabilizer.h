// -*- coding: utf-8 -*-
#ifndef STABILIZER_H
#define STABILIZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

/**
 * @brief Video stabilization class using optical flow (no videostab module required)
 */
class Stabilizer {
public:
    Stabilizer();
    ~Stabilizer();

    bool init(const std::string& input_path);
    cv::Mat next_frame();
    double get_fps() const;
    int get_frame_count() const;

private:
    cv::VideoCapture cap;
    cv::Mat prevGray;
    bool initialized;
};

#endif // STABILIZER_H
