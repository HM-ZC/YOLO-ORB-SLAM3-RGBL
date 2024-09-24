//
// Created by yuwenlu on 2022/3/14.
//
#include <YoloDetect.h>

YoloDetection::YoloDetection()
{
    torch::jit::setTensorExprFuserEnabled(false);
    mModule = torch::jit::load("/root/YOLO_ORB_SLAM3/yolov5s.torchscript");

    std::ifstream f("/root/YOLO_ORB_SLAM3/coco.names");
    std::string name = "";
    while (std::getline(f, name))
    {
        mClassnames.push_back(name);
    }
    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
                      "dog", "horse", "sheep", "crow", "bear"};
}

YoloDetection::~YoloDetection()
{

}

bool YoloDetection::Detect()
{
    cv::Mat img;

    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!" << std::endl;
        return false;
    }

    // Preparing input tensor
    cv::resize(mRGB, img, cv::Size(640, 640));  // Resize the image to the model input size
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);  // Convert BGR to RGB format
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    imgTensor = imgTensor.permute({2, 0, 1});  // Rearrange dimensions to [3, H, W]
    imgTensor = imgTensor.toType(torch::kFloat);  // Convert to float
    imgTensor = imgTensor.div(255);  // Normalize to [0, 1]
    imgTensor = imgTensor.unsqueeze(0);  // Add batch dimension [1, 3, H, W]

    torch::Tensor preds = mModule.forward({imgTensor}).toTuple()->elements()[0].toTensor();  // Forward pass
    std::vector<torch::Tensor> dets = YoloDetection::non_max_suppression(preds, 0.4, 0.5);  // Perform NMS

    if (dets.size() > 0)
    {
        // Visualize result
        for (size_t i = 0; i < dets[0].sizes()[0]; ++i)
        {
            // Extract bounding box and class ID
            float left = dets[0][i][0].item().toFloat() * mRGB.cols / 640;
            float top = dets[0][i][1].item().toFloat() * mRGB.rows / 384;
            float right = dets[0][i][2].item().toFloat() * mRGB.cols / 640;
            float bottom = dets[0][i][3].item().toFloat() * mRGB.rows / 384;
            int classID = dets[0][i][5].item().toInt();
            float confidence = dets[0][i][4].item().toFloat();

            // Draw bounding box
            cv::Rect2i DetectArea(left, top, (right - left), (bottom - top));
            cv::rectangle(mRGB, DetectArea, cv::Scalar(0, 255, 0), 2);  // Green box for detected objects

            // Put class name and confidence score on top of the box
            std::string label = mClassnames[classID] + " " + cv::format("%.2f", confidence);
            int baseline = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            top = std::max(top, (float)labelSize.height);
            cv::rectangle(mRGB, cv::Point(left, top - labelSize.height),
                          cv::Point(left + labelSize.width, top + baseline),
                          cv::Scalar(0, 255, 0), cv::FILLED);  // Green background for label text
            cv::putText(mRGB, label, cv::Point(left, top),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);  // Black text for class name and confidence

            // If it's a dynamic object, add it to the dynamic area list
            if (count(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[classID]))
            {
                mvDynamicArea.push_back(DetectArea);
            }
        }

        if (mvDynamicArea.size() == 0)
        {
            // Default dynamic area if no dynamic object is detected
            cv::Rect2i tDynamicArea(1, 1, 1, 1);
            mvDynamicArea.push_back(tDynamicArea);
        }
    }

    // Show the image with bounding boxes
    cv::imshow("YOLO Detection", mRGB);
    cv::waitKey(1);  // Display the image for a short time, e.g., 1ms

    return true;
}


vector<torch::Tensor> YoloDetection::non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh)
{
    std::vector<torch::Tensor> output;
    for (size_t i=0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>( torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0) continue;

        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor  dets = pred.slice(1, 0, 6);

        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (size_t i=0; i<indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;


            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
}

void YoloDetection::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}

void YoloDetection::ClearImage()
{
    mRGB = 0;
}

void YoloDetection::ClearArea()
{
    mvPersonArea.clear();
}