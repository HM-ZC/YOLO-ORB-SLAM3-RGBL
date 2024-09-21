/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include<opencv2/core/core.hpp>

#include"../../../include/System.h"

#include <opencv2/core/eigen.hpp>
#include <Eigen/Geometry>
using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM):mpSLAM(pSLAM){}

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);

    ORB_SLAM3::System* mpSLAM;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 Mono path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }

    // 创建 YOLO 检测器
    YoloDetection* pYoloDetector = new YoloDetection();

    // 初始化 SLAM 系统
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);

    // 获取 Tracking 对象
    ORB_SLAM3::Tracking* pTracking = SLAM.GetTracker();

    // 设置 YOLO 检测器到 Tracking 中
    if (pTracking != nullptr)
    {
        pTracking->SetDetector(pYoloDetector);
    }

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nodeHandler;
    ros::Subscriber sub = nodeHandler.subscribe("/usb_cam/image_raw", 1, &ImageGrabber::GrabImage, &igb);

    ros::spin();

    // 关闭 SLAM 系统
    SLAM.Shutdown();

    // 保存相机轨迹
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    // 将ROS图像消息转换为cv::Mat格式
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // 执行SLAM跟踪，获取相机的位姿Sophus::SE3f对象
    Sophus::SE3f Tcw_SE3 = mpSLAM->TrackMonocular(cv_ptr->image, cv_ptr->header.stamp.toSec());

    // 将SE3对象转换为Eigen的4x4矩阵
    Eigen::Matrix4f Tcw_matrix = Tcw_SE3.matrix();

    // 将 Eigen::Matrix4f 转换为 Eigen::Matrix<double, 4, 4>
    Eigen::Matrix4d Tcw_matrix_double = Tcw_matrix.cast<double>();

    // 如果没有足够的关键点，返回
    if (vDynamicKeyPoints.size() < 6)
    {
        std::cout << "Not enough keypoints for PnP." << std::endl;
    }
    else
    {
    // 获取当前帧的地图点，并将关键点与地图点匹配，形成二维-三维点对
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> objectPoints;

    ORB_SLAM3::Frame& currentFrame = mpSLAM->GetTracker()->mCurrentFrame;
    for (auto& keypoint : vDynamicKeyPoints)
    {
        // 遍历当前帧中的关键点，找到匹配的地图点
        int idx = -1;
        for (int i = 0; i < currentFrame.N; ++i)  // 修改 size_t 为 int 以避免警告
        {
            if (cv::norm(currentFrame.mvKeysUn[i].pt - keypoint.pt) < 3.0)
            {
                idx = i;
                break;
            }
        }

        if (idx >= 0 && currentFrame.mvpMapPoints[idx])
        {
            ORB_SLAM3::MapPoint* pMP = currentFrame.mvpMapPoints[idx];
            if (pMP)
            {
                // 获取地图点的三维世界坐标 (Eigen::Vector3f)
                Eigen::Vector3f Pw_eigen = pMP->GetWorldPos();
                
                // 将 Eigen::Vector3f 转换为 cv::Point3f
                cv::Point3f Pw(Pw_eigen[0], Pw_eigen[1], Pw_eigen[2]);

                // 添加到二维-三维点对中
                imagePoints.push_back(keypoint.pt);
                objectPoints.push_back(Pw);
            }
        }
    }
    vDynamicKeyPoints.clear();
    // 检查是否有足够的匹配点
    if (imagePoints.size() < 6)
    {
        std::cout << "Not enough matches for PnP." << std::endl;
    }
    else
    {
    // 定义相机内参矩阵
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        469.8, 0, 334.8,
        0, 469.8, 240.2,
        0, 0, 1);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << -0.0555, 0.0907, 0.0, 0.0, 0.0);

    // 使用 solvePnP 计算物体的位姿（相对于相机）
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    if (success)
    {
        // 将旋转向量转换为旋转矩阵
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // 将相机坐标系下的位姿转换为4x4的变换矩阵
        cv::Mat Tco = cv::Mat::eye(4, 4, CV_64F);  // 确保为CV_64F类型
        R.copyTo(Tco(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(Tco(cv::Rect(3, 0, 1, 3)));

        // 将物体的相机坐标系位姿转换为世界坐标系
        cv::Mat Tcw_cv;
        cv::eigen2cv(Tcw_matrix_double, Tcw_cv);  // 将 Eigen::Matrix<double> 转换为 cv::Mat

        // 确保所有矩阵为 CV_64F 类型
        Tcw_cv.convertTo(Tcw_cv, CV_64F);
        Tco.convertTo(Tco, CV_64F);

        // 矩阵乘法，计算世界坐标下的物体位姿
        cv::Mat Tow = Tcw_cv * Tco;

        // 提取物体的平移向量
        cv::Mat objectWorldPos = Tow(cv::Rect(3, 0, 1, 3));
    // 物体相对于相机坐标系的平移向量（tvec 就是相机坐标系下的物体位置）
        std::cout << "物体的相机坐标: " << tvec.t() << std::endl;
        // 输出物体的世界坐标位置
        std::cout << "物体世界坐标: " << objectWorldPos.t() << std::endl;
    }
    else
    {
        std::cout << "PnP computation failed." << std::endl;
    }
    }
    }
    // 输出相机的位置信息和欧拉角
    Eigen::Matrix3f rotation_matrix = Tcw_SE3.so3().matrix();
    Eigen::Vector3f euler_angles = rotation_matrix.eulerAngles(2, 1, 0);  // ZYX 顺序的欧拉角

    float yaw = euler_angles[0] * 180.0 / M_PI;
    float pitch = euler_angles[1] * 180.0 / M_PI;
    float roll = euler_angles[2] * 180.0 / M_PI;

    // 输出相机的位置信息和欧拉角
    cv::Mat Tcw_cv;
    cv::eigen2cv(Tcw_matrix_double, Tcw_cv);  // 将 Eigen::Matrix<double> 转换为 cv::Mat
    Tcw_cv.convertTo(Tcw_cv, CV_64F);  // 确保类型一致

    cv::Mat Twc;
    cv::invert(Tcw_cv, Twc);  // 反转矩阵

    cv::Mat cameraWorldPos = Twc(cv::Rect(3, 0, 1, 3));

    std::cout << "Camera World Coordinates: " << cameraWorldPos.t() << std::endl;
    std::cout << "Camera Euler Angles (Yaw, Pitch, Roll): " 
              << yaw << "°, " << pitch << "°, " << roll << "°" << std::endl;
}