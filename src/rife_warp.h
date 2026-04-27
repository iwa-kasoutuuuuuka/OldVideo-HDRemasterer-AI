#ifndef RIFE_WARP_H
#define RIFE_WARP_H

#include "layer.h"

class Warp : public ncnn::Layer
{
public:
    Warp();
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
};

ncnn::Layer* Warp_layer_creator(void*);

#endif // RIFE_WARP_H
