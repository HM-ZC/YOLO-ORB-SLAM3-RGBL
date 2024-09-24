//
// Created by yuwenlu on 2022/3/14.
//
#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <map>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

// 声明全局变量，其他文件包含此头文件后可以使用这些变量
extern vector<cv::Rect2i> gPersonArea;    // 存储检测到的人物区域
extern vector<cv::Rect2i> gDynamicArea;   // 存储检测到的动态对象区域

class YoloDetection
{
public:
    YoloDetection();
    ~YoloDetection();
    void GetImage(const cv::Mat& RGB); // 修改为接受 const 引用
    void ClearImage();
    bool Detect();
    void ClearArea();
    vector<cv::Rect2i> mvPersonArea; // 存储检测到的人物区域
    vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5);

public:
    cv::Mat mRGB; // 输入图像
    torch::jit::script::Module mModule; // TorchScript模型
    std::vector<std::string> mClassnames; // 类别名称

    // 动态对象相关
    vector<string> mvDynamicNames;
    vector<cv::Rect2i> mvDynamicArea; // 存储检测到的动态对象区域
    map<string, vector<cv::Rect2i>> mmDetectMap;
};

#endif //YOLO_DETECT_H