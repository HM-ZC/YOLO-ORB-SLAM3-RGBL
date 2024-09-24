//
// Created by yuwenlu on 2022/3/14.
//
#include "YoloDetect.h"

// 定义全局变量
vector<cv::Rect2i> gPersonArea;
vector<cv::Rect2i> gDynamicArea;

YoloDetection::YoloDetection()
{
    // 禁用Tensor Expression Fuser以避免潜在的问题
    torch::jit::setTensorExprFuserEnabled(false);

    // 加载TorchScript模型
    mModule = torch::jit::load("/root/YOLO_ORB_SLAM3/yolov5s.torchscript");

    // 读取类别名称
    std::ifstream f("/root/YOLO_ORB_SLAM3/coco.names");
    std::string name = "";
    while (std::getline(f, name))
    {
        mClassnames.push_back(name);
    }

    // 定义动态对象的类别名称
    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", 
                     "bird", "cat", "dog", "horse", "sheep", "crow", "bear"};
}

YoloDetection::~YoloDetection()
{
    // 析构函数，可以在这里释放资源
}

void YoloDetection::GetImage(const cv::Mat& RGB)
{
    mRGB = RGB.clone(); // 复制图像以避免引用同一内存
}

void YoloDetection::ClearImage()
{
    mRGB.release(); // 释放图像内存
}

void YoloDetection::ClearArea()
{
    mvPersonArea.clear();
    mvDynamicArea.clear();
    // 同时清空全局变量
    gPersonArea.clear();
    gDynamicArea.clear();
}

bool YoloDetection::Detect()
{
    cv::Mat img;

    // 检查输入图像是否为空
    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!" << std::endl;
        return false;
    }

    // 准备输入张量
    cv::resize(mRGB, img, cv::Size(640, 640));  // 调整图像大小为模型输入尺寸
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);   // 将BGR格式转换为RGB格式
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte).clone();
    imgTensor = imgTensor.permute({2, 0, 1});    // 调整维度为[3, H, W]
    imgTensor = imgTensor.toType(torch::kFloat); // 转换为浮点型
    imgTensor = imgTensor.div(255);               // 归一化到[0, 1]
    imgTensor = imgTensor.unsqueeze(0);           // 添加批次维度[1, 3, H, W]

    // 前向传播
    torch::Tensor preds = mModule.forward({imgTensor}).toTuple()->elements()[0].toTensor();
    // 执行非极大值抑制
    std::vector<torch::Tensor> dets = YoloDetection::non_max_suppression(preds, 0.4, 0.5);

    // 清空之前的全局区域
    gPersonArea.clear();
    gDynamicArea.clear();
    mvPersonArea.clear();
    mvDynamicArea.clear();

    if (dets.size() > 0 && dets[0].sizes()[0] > 0)
    {
        // 遍历检测结果
        for (size_t i = 0; i < dets[0].sizes()[0]; ++i)
        {
            // 提取边框和类别ID
            float left = dets[0][i][0].item().toFloat() * mRGB.cols / 640;
            float top = dets[0][i][1].item().toFloat() * mRGB.rows / 640;
            float right = dets[0][i][2].item().toFloat() * mRGB.cols / 640;
            float bottom = dets[0][i][3].item().toFloat() * mRGB.rows / 640;
            int classID = dets[0][i][5].item().toInt();
            float confidence = dets[0][i][4].item().toFloat();

            // 创建检测区域矩形
            cv::Rect2i DetectArea(left, top, (right - left), (bottom - top));
            // 绘制绿色边框
            cv::rectangle(mRGB, DetectArea, cv::Scalar(0, 255, 0), 2);

            // 在边框上方显示类别名称和置信度
            std::string label = mClassnames[classID] + " " + cv::format("%.2f", confidence);
            int baseline = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            top = std::max(top, (float)labelSize.height);
            cv::rectangle(mRGB, cv::Point(left, top - labelSize.height),
                          cv::Point(left + labelSize.width, top + baseline),
                          cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(mRGB, label, cv::Point(left, top),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

            // 如果是动态对象，添加到动态区域列表
            if (std::find(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[classID]) != mvDynamicNames.end())
            {
                mvDynamicArea.push_back(DetectArea);
                gDynamicArea.push_back(DetectArea); // 保存到全局变量
            }

            // 如果是人物，添加到人物区域列表
            if (mClassnames[classID] == "person")
            {
                mvPersonArea.push_back(DetectArea);
                gPersonArea.push_back(DetectArea); // 保存到全局变量
            }
        }

        // 如果没有检测到动态对象，添加默认区域
        if (mvDynamicArea.empty())
        {
            cv::Rect2i tDynamicArea(1, 1, 1, 1);
            mvDynamicArea.push_back(tDynamicArea);
            gDynamicArea.push_back(tDynamicArea);
        }
    }

    // 显示带有检测框的图像
    cv::imshow("YOLO Detection", mRGB);
    cv::waitKey(1); // 等待1毫秒

    return true;
}

vector<torch::Tensor> YoloDetection::non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh)
{
    std::vector<torch::Tensor> output;
    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        // 按置信度筛选
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = pred.index_select(0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0) continue;

        // 将边框从中心点格式转换为左上右下格式
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // 计算最终的置信度和类别
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor dets = pred.slice(1, 0, 6);

        // 计算每个检测框的面积
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));

        // 按置信度排序
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, /*descending=*/true);
        torch::Tensor scores_sorted = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);

        std::vector<int64_t> keep_indices;

        while (indexes.size(0) > 0)
        {
            // 保留当前最高分的检测框
            int64_t current = indexes[0].item().toInt();
            keep_indices.push_back(current);

            if (indexes.size(0) == 1)
                break;

            // 计算当前检测框与剩余检测框的IOU
            torch::Tensor current_box = dets.select(0, current).slice(0, 0, 4); // [left, top, right, bottom]
            torch::Tensor other_boxes = dets.index({indexes.slice(0, 1, indexes.size(0))}).slice(1, 0, 4); // [N, 4]

            // 计算交集
            torch::Tensor inter_left = torch::maximum(current_box[0], other_boxes.select(1, 0));
            torch::Tensor inter_top = torch::maximum(current_box[1], other_boxes.select(1, 1));
            torch::Tensor inter_right = torch::minimum(current_box[2], other_boxes.select(1, 2));
            torch::Tensor inter_bottom = torch::minimum(current_box[3], other_boxes.select(1, 3));

            torch::Tensor inter_width = (inter_right - inter_left).clamp(0);
            torch::Tensor inter_height = (inter_bottom - inter_top).clamp(0);
            torch::Tensor inter_area = inter_width * inter_height;

            // 计算IOU
            torch::Tensor iou = inter_area / (areas[current] + areas.index_select(0, indexes.slice(0, 1, indexes.size(0))) - inter_area);

            // 保留IOU小于阈值的检测框
            torch::Tensor mask = (iou <= iou_thresh).nonzero().squeeze();

            if (mask.numel() == 0)
                break;

            // 更新索引，只保留IOU小于阈值的框
            indexes = indexes.index_select(0, mask + 1);
        }

        // 将保留的索引转换为Tensor并添加到输出
        if (!keep_indices.empty())
        {
            torch::Tensor keep = torch::from_blob(keep_indices.data(), {static_cast<long>(keep_indices.size())}, torch::kInt64).clone();
            output.push_back(dets.index_select(0, keep));
        }
    }
    return output;
}
