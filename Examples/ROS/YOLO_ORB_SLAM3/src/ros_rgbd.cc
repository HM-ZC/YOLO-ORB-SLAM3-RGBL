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


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <map>
#include <vector>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>

#include "../../../include/System.h"
#include <opencv2/core/eigen.hpp>
#include <Eigen/Geometry>

#include "YoloDetect.h"

using namespace std;

// 全局变量声明
extern vector<cv::Rect2i> gPersonArea;
extern vector<cv::Rect2i> gDynamicArea;

// 结构体用于存储相机位姿信息
struct Pose
{
    Eigen::Vector3d position;         // 相机位置（世界坐标系）
    Eigen::Matrix3d orientation;      // 相机朝向（旋转矩阵）
    ros::Time timestamp;              // 时间戳
};

// 结构体用于存储每个人物的信息
struct PersonInfo
{
    Eigen::Vector3d position;              // 人物位置（相对相机）
    ros::Time timestamp;                   // 时间戳
    Eigen::Vector3d velocity;              // 人物绝对速度
    Eigen::Vector3d acceleration;          // 人物绝对加速度
    Eigen::Vector3d relativeVelocity;      // 人物相对相机的速度
    Eigen::Vector3d relativeAcceleration;  // 人物相对相机的加速度
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, YoloDetection* pYolo)
        : mpSLAM(pSLAM), mpYoloDetector(pYolo)
    {
        // 初始化相机位姿
        previousCameraPose.position = Eigen::Vector3d::Zero();
        previousCameraPose.orientation = Eigen::Matrix3d::Identity();
        previousCameraPose.timestamp = ros::Time::now();

        // 初始化速度与加速度
        cameraVelocity = Eigen::Vector3d::Zero();
        previousCameraVelocity = Eigen::Vector3d::Zero();
        cameraAcceleration = Eigen::Vector3d::Zero();
    }

    // 回调函数，处理同步后的RGB和深度图像
    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD);

private:
    ORB_SLAM3::System* mpSLAM;
    YoloDetection* mpYoloDetector; // YOLO检测器指针

    // 相机位姿历史记录
    Pose previousCameraPose;
    Eigen::Vector3d cameraVelocity;
    Eigen::Vector3d previousCameraVelocity;
    Eigen::Vector3d cameraAcceleration;

    // 人物历史记录：映射每个人物ID到其信息
    // 由于YOLO不提供ID，这里简化为基于索引的映射
    // 在实际应用中，建议使用更复杂的追踪算法
    map<int, PersonInfo> previousPersons;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 RGBD path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }    

    // 初始化YOLO检测器
    YoloDetection* pYoloDetector = new YoloDetection();

    // 创建SLAM系统，初始化所有系统线程并准备处理帧
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);
    ORB_SLAM3::Tracking* pTracking = SLAM.GetTracker();
    if (pTracking != nullptr)
    {
        pTracking->SetDetector(pYoloDetector);
    }

    // 创建ImageGrabber对象，并传入SLAM系统和YOLO检测器
    ImageGrabber igb(&SLAM, pYoloDetector);

    ros::NodeHandle nh;

    // 订阅RGB和深度图像话题
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_rect_raw", 1);

    // 设置同步策略
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub, depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD, &igb, _1, _2));

    ros::spin();

    // 停止SLAM系统
    SLAM.Shutdown();

    // 保存相机轨迹
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD)
{
    // 将ROS图像消息转换为cv::Mat格式
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // 更新YOLO检测器的图像并进行检测
    mpYoloDetector->GetImage(cv_ptrRGB->image);
    mpYoloDetector->Detect();

    // 获取SLAM的位姿
    Sophus::SE3f Tcw_SE3 = mpSLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, cv_ptrRGB->header.stamp.toSec());
    // 将SE3对象转换为Eigen的4x4矩阵
    Eigen::Matrix4f Tcw_matrix = Tcw_SE3.matrix();

    // 获取当前时间
    ros::Time currentTime = msgRGB->header.stamp;

    // 相机内参（请根据实际相机参数进行调整）
    double fx = 615.304;
    double fy = 615.590;
    double cx = 326.230;
    double cy = 240.635;

    // 提取相机位置和朝向
    Eigen::Vector3d currentPosition(Tcw_matrix(0,3), Tcw_matrix(1,3), Tcw_matrix(2,3));
    // 使用 .cast<double>() 显式转换为 double 类型
    Eigen::Matrix3d currentOrientation = Tcw_matrix.block<3,3>(0,0).cast<double>();

    // 计算相机的速度和加速度
    double dt = (currentTime - previousCameraPose.timestamp).toSec();
    if(dt > 0.0)
    {
        // 计算速度
        cameraVelocity = (currentPosition - previousCameraPose.position) / dt;
        // 计算加速度
        cameraAcceleration = (cameraVelocity - previousCameraVelocity) / dt;
    }
    else
    {
        // 如果是初始帧，没有前一帧数据
        cameraVelocity = Eigen::Vector3d::Zero();
        cameraAcceleration = Eigen::Vector3d::Zero();
    }

    // 更新前一帧的速度
    previousCameraVelocity = cameraVelocity;

    // 更新前一帧的位姿信息
    previousCameraPose.position = currentPosition;
    previousCameraPose.orientation = currentOrientation;
    previousCameraPose.timestamp = currentTime;

    // 获取相机的欧拉角（ZYX顺序）
    Eigen::Vector3d euler_angles = currentOrientation.eulerAngles(2, 1, 0);  // ZYX 顺序的欧拉角
    double yaw = euler_angles[0] * 180.0 / M_PI;
    double pitch = euler_angles[1] * 180.0 / M_PI;
    double roll = euler_angles[2] * 180.0 / M_PI;

    // 输出相机的位置信息、速度、加速度和欧拉角
    std::cout << "========================================" << std::endl;
    std::cout << "相机信息:" << std::endl;
    std::cout << "相机世界坐标: [" << currentPosition(0) << ", " 
              << currentPosition(1) << ", " << currentPosition(2) << "] m" << std::endl;
    std::cout << "相机相对世界的速度: [" << cameraVelocity(0) << ", " 
              << cameraVelocity(1) << ", " << cameraVelocity(2) << "] m/s" << std::endl;
    std::cout << "相机相对世界的加速度: [" << cameraAcceleration(0) << ", " 
              << cameraAcceleration(1) << ", " << cameraAcceleration(2) << "] m/s²" << std::endl;
    std::cout << "相机 Euler Angles (Yaw, Pitch, Roll): " 
              << yaw << "°, " << pitch << "°, " << roll << "°" << std::endl;

    // 遍历所有检测到的人物边框
    int person_id = 0;
    for(const auto& rect : gPersonArea)
    {
        // 计算边框中心点
        int center_x = rect.x + rect.width / 2;
        int center_y = rect.y + rect.height / 2;

        // 获取中心点的深度值
        float depth = cv_ptrD->image.at<float>(center_y, center_x);

        // 如果深度值无效，尝试在边框内搜索有效深度
        if(std::isnan(depth) || depth <= 0.0)
        {
            bool found = false;
            for(int dy = -5; dy <= 5 && !found; dy++)
            {
                for(int dx = -5; dx <= 5 && !found; dx++)
                {
                    int nx = center_x + dx;
                    int ny = center_y + dy;
                    if(nx >= 0 && nx < cv_ptrD->image.cols && ny >= 0 && ny < cv_ptrD->image.rows)
                    {
                        depth = cv_ptrD->image.at<float>(ny, nx);
                        if(!std::isnan(depth) && depth > 0.0)
                        {
                            found = true;
                        }
                    }
                }
            }
            if(!found)
            {
                std::cout << "========================================" << std::endl;
                std::cout << "人物 " << person_id << ": 无法获取有效的深度值，跳过该人物。" << std::endl;
                person_id++;
                continue;
            }
        }

        // 将像素坐标和深度转换为相机坐标系下的3D点
        double X = (center_x - cx) * depth / fx;
        double Y = (center_y - cy) * depth / fy;
        double Z = depth;

        // 构建相机坐标系下的向量
        Eigen::Vector4d Pc(X, Y, Z, 1.0);

        // 转换为世界坐标系下的坐标
        Eigen::Matrix4d Tcw_double = Tcw_matrix.cast<double>();
        Eigen::Vector4d Pw = Tcw_double * Pc;

        Eigen::Vector3d personWorldPosition = Pw.head<3>();

        // 计算相对于相机的坐标（直接使用 Pc）
        Eigen::Vector3d relativePosition = Pc.head<3>();

        // 输出人物的世界坐标
        std::cout << "========================================" << std::endl;
        std::cout << "人物 " << person_id << " 信息:" << std::endl;
        std::cout << "  人物世界坐标: [" << Pw(0) << ", " << Pw(1) << ", " << Pw(2) << "] m" << std::endl;
        std::cout << "  人物相对相机的坐标: [" << relativePosition(0) << ", " 
                  << relativePosition(1) << ", " << relativePosition(2) << "] m" << std::endl;

        // 假设人物的角度与相机一致（缺乏具体数据，或者需要额外的姿态估计）
        // 这里只输出相机的角度作为示例
        Eigen::Vector3d relativeEulerAngles = currentOrientation.eulerAngles(2, 1, 0); // ZYX 顺序的欧拉角
        relativeEulerAngles = relativeEulerAngles * 180.0 / M_PI;
        std::cout << "  人物相对相机的角度 (Yaw, Pitch, Roll): [" 
                  << relativeEulerAngles(0) << "°, " 
                  << relativeEulerAngles(1) << "°, " 
                  << relativeEulerAngles(2) << "°]" << std::endl;

        // 计算人物速度和加速度
        // 由于YOLO不提供跟踪ID，这里简化为基于索引的映射
        // 在实际应用中，建议集成更复杂的跟踪算法以准确追踪每个人物
        auto it = previousPersons.find(person_id);
        if(it != previousPersons.end())
        {
            // 计算时间间隔
            double person_dt = (currentTime - it->second.timestamp).toSec();
            if(person_dt > 0.0)
            {
                // 计算速度
                Eigen::Vector3d personVelocity = (relativePosition - it->second.position) / person_dt;
                // 计算加速度
                Eigen::Vector3d personAcceleration = (personVelocity - it->second.velocity) / person_dt;

                // 计算相对速度和加速度
                Eigen::Vector3d relativeVelocity = personVelocity - cameraVelocity;
                Eigen::Vector3d relativeAcceleration = personAcceleration - cameraAcceleration;

                // 更新人物信息
                previousPersons[person_id].velocity = personVelocity;
                previousPersons[person_id].acceleration = personAcceleration;
                previousPersons[person_id].relativeVelocity = relativeVelocity;
                previousPersons[person_id].relativeAcceleration = relativeAcceleration;
                previousPersons[person_id].position = relativePosition;
                previousPersons[person_id].timestamp = currentTime;
                std::cout << "========================================" << std::endl;
                // 输出速度和加速度
                std::cout << "  人物的速度: [" << personVelocity(0) << ", " 
                          << personVelocity(1) << ", " << personVelocity(2) << "] m/s" << std::endl;
                std::cout << "  人物的加速度: [" << personAcceleration(0) << ", " 
                          << personAcceleration(1) << ", " << personAcceleration(2) << "] m/s²" << std::endl;
                std::cout << "  人物相对相机的速度: [" << relativeVelocity(0) << ", " 
                          << relativeVelocity(1) << ", " << relativeVelocity(2) << "] m/s" << std::endl;
                std::cout << "  人物相对相机的加速度: [" << relativeAcceleration(0) << ", " 
                          << relativeAcceleration(1) << ", " << relativeAcceleration(2) << "] m/s²" << std::endl;
            }
            else
            {
                // 时间间隔为0，无法计算速度和加速度
                std::cout << "  人物相对相机的速度: 无法计算 (时间间隔为0)" << std::endl;
                std::cout << "  人物相对相机的加速度: 无法计算 (时间间隔为0)" << std::endl;
            }
        }
        else
        {
            // 首次检测到该人物，初始化其信息
            PersonInfo info;
            info.position = relativePosition;
            info.velocity = Eigen::Vector3d::Zero();
            info.acceleration = Eigen::Vector3d::Zero();
            info.relativeVelocity = Eigen::Vector3d::Zero();
            info.relativeAcceleration = Eigen::Vector3d::Zero();
            info.timestamp = currentTime;
            previousPersons[person_id] = info;
        }

        person_id++;
    }
}