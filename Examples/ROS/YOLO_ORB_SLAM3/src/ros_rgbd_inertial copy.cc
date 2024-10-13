#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <map>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Accel.h>

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
    Eigen::Vector3d Pw;
    Eigen::Vector3d position;              // 人物位置（相对相机）
    ros::Time timestamp;                   // 时间戳
    Eigen::Vector3d velocity;              // 人物绝对速度
    Eigen::Vector3d acceleration;          // 人物绝对加速度
    Eigen::Vector3d relativeVelocity;      // 人物相对相机的速度
    Eigen::Vector3d relativeAcceleration;  // 人物相对相机的加速度
};

// IMU数据接收类
class ImuGrabber
{
public:
    ImuGrabber() {}

// 回调函数，接收IMU数据并存入缓冲区
void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 打印IMU消息的内容
        ROS_INFO("接收到的IMU数据:");
        ROS_INFO("  角速度: [%f, %f, %f]", 
                 imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
        ROS_INFO("  线性加速度: [%f, %f, %f]", 
                 imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);

    if (std::isnan(imu_msg->angular_velocity.x) || std::isnan(imu_msg->angular_velocity.y) || std::isnan(imu_msg->angular_velocity.z) ||
        std::isnan(imu_msg->linear_acceleration.x) || std::isnan(imu_msg->linear_acceleration.y) || std::isnan(imu_msg->linear_acceleration.z))
    {
        ROS_WARN("IMU数据包含NaN值，丢弃该帧！");
        return;  // 丢弃该帧
    }

    std::lock_guard<std::mutex> lock(mBufMutex);
    imuBuf.push(imu_msg);
}

    // IMU数据缓冲区
    std::queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

// 图像和IMU数据处理类
class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, YoloDetection* pYolo, ImuGrabber* pImuGb)
        : mpSLAM(pSLAM), mpYoloDetector(pYolo), mpImuGb(pImuGb)
    {
        // 初始化相机位姿
        previousCameraPose.position = Eigen::Vector3d::Zero();
        previousCameraPose.orientation = Eigen::Matrix3d::Identity();
        previousCameraPose.timestamp = ros::Time::now();

        // 初始化速度与加速度
        cameraVelocity = Eigen::Vector3d::Zero();
        previousCameraVelocity = Eigen::Vector3d::Zero();
        cameraAcceleration = Eigen::Vector3d::Zero();

        // 初始化ROS发布者
        camera_pose_pub = nh.advertise<geometry_msgs::Pose>("camera_pose", 10);
        camera_velocity_pub = nh.advertise<geometry_msgs::Twist>("camera_velocity", 10);
        camera_acceleration_pub = nh.advertise<geometry_msgs::Accel>("camera_acceleration", 10);

        person_pose_pub = nh.advertise<geometry_msgs::Pose>("person_pose", 10);
        person_velocity_pub = nh.advertise<geometry_msgs::Twist>("person_velocity", 10);
        person_acceleration_pub = nh.advertise<geometry_msgs::Accel>("person_acceleration", 10);

        // 发布人物相对相机的坐标、速度和加速度
        relative_person_pose_pub = nh.advertise<geometry_msgs::Pose>("relative_person_pose", 10);
        relative_person_velocity_pub = nh.advertise<geometry_msgs::Twist>("relative_person_velocity", 10);
        relative_person_acceleration_pub = nh.advertise<geometry_msgs::Accel>("relative_person_acceleration", 10);
    }
    void PublishCameraInfo()
    {
        // 发布相机的位置信息（使用 Pose）
        geometry_msgs::Pose camera_pose_msg;
        camera_pose_msg.position.x = previousCameraPose.position(0);
        camera_pose_msg.position.y = previousCameraPose.position(1);
        camera_pose_msg.position.z = previousCameraPose.position(2);

        Eigen::Quaterniond q(previousCameraPose.orientation);
        camera_pose_msg.orientation.x = q.x();
        camera_pose_msg.orientation.y = q.y();
        camera_pose_msg.orientation.z = q.z();
        camera_pose_msg.orientation.w = q.w();

        camera_pose_pub.publish(camera_pose_msg);

        // 发布相机的速度（使用 Twist）
        geometry_msgs::Twist camera_velocity_msg;
        camera_velocity_msg.linear.x = cameraVelocity(0);
        camera_velocity_msg.linear.y = cameraVelocity(1);
        camera_velocity_msg.linear.z = cameraVelocity(2);

        camera_velocity_pub.publish(camera_velocity_msg);

        // 发布相机的加速度（使用 Accel）
        geometry_msgs::Accel camera_acceleration_msg;
        camera_acceleration_msg.linear.x = cameraAcceleration(0);
        camera_acceleration_msg.linear.y = cameraAcceleration(1);
        camera_acceleration_msg.linear.z = cameraAcceleration(2);

        camera_acceleration_pub.publish(camera_acceleration_msg);
    }

    void PublishPersonInfo(int person_id, const PersonInfo& person_info)
    {
        // 发布人物的位置信息（使用 Pose）
        geometry_msgs::Pose person_pose_msg;
        person_pose_msg.position.x = person_info.Pw(0);
        person_pose_msg.position.y = person_info.Pw(1);
        person_pose_msg.position.z = person_info.Pw(2);

        // 假设人物的方向与相机一致，因此可以复用相机的姿态（这可以根据实际情况调整）
        Eigen::Quaterniond q(previousCameraPose.orientation);
        person_pose_msg.orientation.x = q.x();
        person_pose_msg.orientation.y = q.y();
        person_pose_msg.orientation.z = q.z();
        person_pose_msg.orientation.w = q.w();

        person_pose_pub.publish(person_pose_msg);

        // 发布人物的速度（使用 Twist）
        geometry_msgs::Twist person_velocity_msg;
        person_velocity_msg.linear.x = person_info.velocity(0);
        person_velocity_msg.linear.y = person_info.velocity(1);
        person_velocity_msg.linear.z = person_info.velocity(2);

        person_velocity_pub.publish(person_velocity_msg);

        // 发布人物的加速度（使用 Accel）
        geometry_msgs::Accel person_acceleration_msg;
        person_acceleration_msg.linear.x = person_info.acceleration(0);
        person_acceleration_msg.linear.y = person_info.acceleration(1);
        person_acceleration_msg.linear.z = person_info.acceleration(2);

        person_acceleration_pub.publish(person_acceleration_msg);
        // 发布人物相对相机的位置信息（使用 Pose）
        geometry_msgs::Pose relative_person_pose_msg;
        relative_person_pose_msg.position.x = person_info.position(0);
        relative_person_pose_msg.position.y = person_info.position(1);
        relative_person_pose_msg.position.z = person_info.position(2);

        relative_person_pose_pub.publish(relative_person_pose_msg);

        // 发布人物相对相机的速度（使用 Twist）
        geometry_msgs::Twist relative_person_velocity_msg;
        relative_person_velocity_msg.linear.x = person_info.relativeVelocity(0);
        relative_person_velocity_msg.linear.y = person_info.relativeVelocity(1);
        relative_person_velocity_msg.linear.z = person_info.relativeVelocity(2);

        relative_person_velocity_pub.publish(relative_person_velocity_msg);

        // 发布人物相对相机的加速度（使用 Accel）
        geometry_msgs::Accel relative_person_acceleration_msg;
        relative_person_acceleration_msg.linear.x = person_info.relativeAcceleration(0);
        relative_person_acceleration_msg.linear.y = person_info.relativeAcceleration(1);
        relative_person_acceleration_msg.linear.z = person_info.relativeAcceleration(2);

        relative_person_acceleration_pub.publish(relative_person_acceleration_msg);
    }

    // 调用此函数来发布相机和人物的信息
    void PublishInfo()
    {
        // 发布相机的速度、加速度和位置信息
        PublishCameraInfo();

        // 发布每个人物的信息
        for (const auto& person : previousPersons)
        {
            PublishPersonInfo(person.first, person.second);
        }
    }
    // 回调函数，处理同步后的RGB和深度图像
    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD)
    {
        // 获取当前时间
        ros::Time currentTime = msgRGB->header.stamp;

        // 将ROS图像消息转换为cv::Mat格式
        cv_bridge::CvImageConstPtr cv_ptrRGB;
        try
        {
            cv_ptrRGB = cv_bridge::toCvShare(msgRGB, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv_bridge::CvImageConstPtr cv_ptrD;
        try
        {
            cv_ptrD = cv_bridge::toCvShare(msgD, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // 更新YOLO检测器的图像并进行检测
        mpYoloDetector->GetImage(cv_ptrRGB->image);
        mpYoloDetector->Detect();
    // 在同步IMU数据之前，丢弃过期的IMU数据
    while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() < currentTime.toSec() - 0.1)
    {
        ROS_WARN("丢弃过期IMU数据，时间差为: %f", currentTime.toSec() - mpImuGb->imuBuf.front()->header.stamp.toSec());
        mpImuGb->imuBuf.pop();
    }

        // 获取IMU数据与当前图像时间戳同步
// 获取IMU数据与当前图像时间戳同步
vector<ORB_SLAM3::IMU::Point> vImuMeas;
{
    std::lock_guard<std::mutex> lock(mpImuGb->mBufMutex);

    // 处理IMU数据
    while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= currentTime.toSec())
    {
        double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
        //ROS_INFO("同步IMU时间: %f, 当前时间: %f", t, currentTime.toSec());
        // 将Eigen::Vector3d 转换为 cv::Point3f
        cv::Point3f acc_cv(
            static_cast<float>(mpImuGb->imuBuf.front()->linear_acceleration.x),
            static_cast<float>(mpImuGb->imuBuf.front()->linear_acceleration.y),
            static_cast<float>(mpImuGb->imuBuf.front()->linear_acceleration.z)
        );
        cv::Point3f gyr_cv(
            static_cast<float>(mpImuGb->imuBuf.front()->angular_velocity.x),
            static_cast<float>(mpImuGb->imuBuf.front()->angular_velocity.y),
            static_cast<float>(mpImuGb->imuBuf.front()->angular_velocity.z)
        );

        vImuMeas.emplace_back(acc_cv, gyr_cv, t);
        mpImuGb->imuBuf.pop();
    }

    // 如果最后一个IMU数据的时间戳比图像时间戳要晚，进行插值
    if (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() > currentTime.toSec())
    {
        // 获取最近的两帧IMU数据用于插值
        ORB_SLAM3::IMU::Point* imu_prev = vImuMeas.empty() ? nullptr : &vImuMeas.back(); // 使用指针获取上一条IMU数据
        sensor_msgs::ImuConstPtr imu_next = mpImuGb->imuBuf.front(); // 下一条IMU数据（未弹出的）

        if (imu_prev)
        {
            double time_diff = imu_next->header.stamp.toSec() - imu_prev->t;  // 使用 t 替代 header.stamp
            if (time_diff > 0)
            {
                double alpha = (currentTime.toSec() - imu_prev->t) / time_diff;  // 使用 t 替代 header.stamp

                // 插值加速度
                cv::Point3f acc_cv_interp(
                    static_cast<float>((1.0 - alpha) * imu_prev->a.x() + alpha * imu_next->linear_acceleration.x),  // 使用 a 替代 linear_acceleration
                    static_cast<float>((1.0 - alpha) * imu_prev->a.y() + alpha * imu_next->linear_acceleration.y),
                    static_cast<float>((1.0 - alpha) * imu_prev->a.z() + alpha * imu_next->linear_acceleration.z)
                );

                // 插值角速度
                cv::Point3f gyr_cv_interp(
                    static_cast<float>((1.0 - alpha) * imu_prev->w.x() + alpha * imu_next->angular_velocity.x),  // 使用 w 替代 angular_velocity
                    static_cast<float>((1.0 - alpha) * imu_prev->w.y() + alpha * imu_next->angular_velocity.y),
                    static_cast<float>((1.0 - alpha) * imu_prev->w.z() + alpha * imu_next->angular_velocity.z)
                );

                vImuMeas.emplace_back(acc_cv_interp, gyr_cv_interp, currentTime.toSec());
            }
        }
    }
}
        // 传递IMU和图像数据到SLAM系统前检查
for (const auto& imu : vImuMeas)
{
    if (std::isnan(imu.a.x()) || std::isnan(imu.a.y()) || std::isnan(imu.a.z()) ||
        std::isnan(imu.w.x()) || std::isnan(imu.w.y()) || std::isnan(imu.w.z()))
    {
        ROS_WARN("SLAM系统中发现无效的IMU数据，跳过该帧！");
        return;
    }
}
        // 将同步后的图像和IMU数据传递给SLAM系统进行处理
        // TrackRGBD 函数返回当前相机位姿
        Sophus::SE3f Tcw_SE3 = mpSLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, currentTime.toSec(), vImuMeas);

        // 将SE3对象转换为Eigen的4x4矩阵
        Eigen::Matrix4f Tcw_matrix = Tcw_SE3.matrix();

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

        // 获取YOLO检测到的人物信息
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
                    previousPersons[person_id].Pw = personWorldPosition;
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
        PublishInfo();
    }

private:
    ros::NodeHandle nh;

    // 发布相机信息的发布者
    ros::Publisher camera_pose_pub;
    ros::Publisher camera_velocity_pub;
    ros::Publisher camera_acceleration_pub;

    // 发布人物信息的发布者
    ros::Publisher person_pose_pub;
    ros::Publisher person_velocity_pub;
    ros::Publisher person_acceleration_pub;

    // 发布人物相对相机的坐标、速度和加速度的发布者
    ros::Publisher relative_person_pose_pub;
    ros::Publisher relative_person_velocity_pub;
    ros::Publisher relative_person_acceleration_pub;

    ORB_SLAM3::System* mpSLAM;
    YoloDetection* mpYoloDetector; // YOLO检测器指针
    ImuGrabber* mpImuGb;           // IMU抓取器指针

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
    ros::init(argc, argv, "RGBD_Inertial");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 RGBD_IMU_YOLO path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }    

    // 初始化YOLO检测器
    YoloDetection* pYoloDetector = new YoloDetection();

    // 创建SLAM系统，初始化所有系统线程并准备处理帧
    // 确保ORB_SLAM3系统支持RGBD和IMU
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_RGBD, true);
    ORB_SLAM3::Tracking* pTracking = SLAM.GetTracker();
    if (pTracking != nullptr)
    {
        pTracking->SetDetector(pYoloDetector);
    }

    // 创建ImuGrabber对象
    ImuGrabber imugb;

    // 创建ImageGrabber对象，并传入SLAM系统、YOLO检测器和IMU抓取器
    ImageGrabber igb(&SLAM, pYoloDetector, &imugb);

    ros::NodeHandle nh;

    // 订阅RGB和深度图像话题
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_rect_raw", 1);

    // 订阅IMU话题
    ros::Subscriber sub_imu = nh.subscribe("/imu", 1000, &ImuGrabber::GrabImu, &imugb);

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

    // 释放YOLO检测器资源
    delete pYoloDetector;

    return 0;
}