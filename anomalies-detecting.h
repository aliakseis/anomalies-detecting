#pragma once

#include "opencv2/core/types.hpp"

cv::Mat getAnomalies(const cv::Mat& frame, const cv::Rect& rect, 
    int squeezeMultiplyFactor // must be a multiple of 4
);
