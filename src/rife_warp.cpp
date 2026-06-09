#include "rife_warp.h"
#include <algorithm>
#include <cmath>

Warp::Warp()
{
    one_blob_only = false;
    support_vulkan = false; // CPU fallback for now
}

int Warp::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    // 入力検証: 不正なブロブ数の場合は早期リターン
    if (bottom_blobs.size() < 2 || top_blobs.size() < 1)
        return -1;

    ncnn::Mat image = bottom_blobs[0];
    ncnn::Mat flow = bottom_blobs[1];

    // flowは最低2チャネル (x, y) が必要
    if (flow.c < 2)
        return -1;

    if (image.elemsize != 4)
    {
        ncnn::cast_float16_to_float32(bottom_blobs[0], image, opt);
    }
    if (flow.elemsize != 4)
    {
        ncnn::cast_float16_to_float32(bottom_blobs[1], flow, opt);
    }

    int w = image.w;
    int h = image.h;
    int channels = image.c;

    ncnn::Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* image_ptr = image.channel(q);
        float* out_ptr = top_blob.channel(q);
        const float* flow_ptr_x = flow.channel(0);
        const float* flow_ptr_y = flow.channel(1);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float dx = flow_ptr_x[y * w + x];
                float dy = flow_ptr_y[y * w + x];

                float src_x = (float)x + dx;
                float src_y = (float)y + dy;

                // NaN または Inf の場合は元の座標に戻してクラッシュを防止
                if (std::isnan(src_x) || std::isinf(src_x)) src_x = (float)x;
                if (std::isnan(src_y) || std::isinf(src_y)) src_y = (float)y;

                if (src_x < 0) src_x = 0;
                if (src_x > (float)w - 1) src_x = (float)w - 1;
                if (src_y < 0) src_y = 0;
                if (src_y > (float)h - 1) src_y = (float)h - 1;

                // Bilinear interpolation
                int x0 = (int)src_x;
                int y0 = (int)src_y;
                int x1 = std::min(x0 + 1, w - 1);
                int y1 = std::min(y0 + 1, h - 1);

                float fx = src_x - (float)x0;
                float fy = src_y - (float)y0;

                float v00 = image_ptr[y0 * w + x0];
                float v01 = image_ptr[y0 * w + x1];
                float v10 = image_ptr[y1 * w + x0];
                float v11 = image_ptr[y1 * w + x1];

                out_ptr[y * w + x] = (1.0f - fx) * (1.0f - fy) * v00 + 
                                     fx * (1.0f - fy) * v01 + 
                                     (1.0f - fx) * fy * v10 + 
                                     fx * fy * v11;
            }
        }
    }

    if (bottom_blobs[0].elemsize != 4)
    {
        ncnn::Mat top_blob_fp16;
        ncnn::cast_float32_to_float16(top_blob, top_blob_fp16, opt);
        top_blob = top_blob_fp16;
    }

    return 0;
}

// 登録用関数マクロ
DEFINE_LAYER_CREATOR(Warp)
