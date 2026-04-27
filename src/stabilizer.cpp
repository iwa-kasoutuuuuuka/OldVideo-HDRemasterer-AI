#include "stabilizer.h"
#include <iostream>
#include <vector>

Stabilizer::Stabilizer() : initialized(false) {}

Stabilizer::~Stabilizer() {
    if (cap.isOpened()) cap.release();
}

bool Stabilizer::init(const std::string& input_path) {
    if (!cap.open(input_path)) {
        std::cerr << "[ERROR] Stabilizer: 動画ファイルを開けません: " << input_path << std::endl;
        return false;
    }

    // 最初のフレームを読み込んでグレースケール化
    cv::Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) {
        std::cerr << "[ERROR] Stabilizer: 最初のフレームが空です。" << std::endl;
        return false;
    }

    cv::cvtColor(firstFrame, prevGray, cv::COLOR_BGR2GRAY);
    cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 先頭に巻き戻し
    initialized = true;

    return true;
}

double Stabilizer::get_fps() const {
    return cap.get(cv::CAP_PROP_FPS);
}

int Stabilizer::get_frame_count() const {
    return static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
}

cv::Mat Stabilizer::next_frame() {
    if (!initialized || !cap.isOpened()) return cv::Mat();

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) return cv::Mat();

    cv::Mat currGray;
    cv::cvtColor(frame, currGray, cv::COLOR_BGR2GRAY);

    // 特徴点の検出 (Good Features to Track)
    std::vector<cv::Point2f> prevPts, currPts;
    cv::goodFeaturesToTrack(prevGray, prevPts, 200, 0.01, 30);

    if (prevPts.empty()) {
        prevGray = currGray.clone();
        return frame;
    }

    // オプティカルフローで特徴点を追跡
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevGray, currGray, prevPts, currPts, status, err);

    // 有効な追跡点のみを抽出
    std::vector<cv::Point2f> validPrev, validCurr;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            validPrev.push_back(prevPts[i]);
            validCurr.push_back(currPts[i]);
        }
    }

    if (validPrev.size() < 10) {
        prevGray = currGray.clone();
        return frame;
    }

    // アフィン変換の推定
    cv::Mat T = cv::estimateAffine2D(validPrev, validCurr);

    if (T.empty()) {
        prevGray = currGray.clone();
        return frame;
    }

    // 逆変換を適用して安定化
    cv::Mat stabilized;
    cv::warpAffine(frame, stabilized, T, frame.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);

    prevGray = currGray.clone();
    return stabilized;
}
