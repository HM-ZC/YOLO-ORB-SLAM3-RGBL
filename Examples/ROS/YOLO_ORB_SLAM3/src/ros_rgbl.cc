#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include"../../../include/System.h"

#include<opencv2/core/eigen.hpp>
#include<Eigen/Geometry>
#include<image_transport/image_transport.h>
#include<sensor_msgs/PointCloud2.h>
#include<pcl_conversions/pcl_conversions.h>
#include<pcl/point_cloud.h>
#include<pcl/point_types.h>
#include<message_filters/subscriber.h>
#include<message_filters/synchronizer.h>
#include<message_filters/sync_policies/approximate_time.h>

#include<geometry_msgs/Point.h>
#include<geometry_msgs/Vector3.h>
#include<std_msgs/Float32.h>
#include<std_msgs/Header.h>

using namespace std;

// 声明全局变量，其他文件包含此头文件后可以使用这些变量
extern vector<cv::Rect2i> gPersonArea;    // 存储检测到的人物区域
extern vector<cv::Rect2i> gDynamicArea;   // 存储检测到的动态对象区域

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM) : mpSLAM(pSLAM), lastTimestamp(-1) {}

    void GrabImagePointCloud(const sensor_msgs::ImageConstPtr& msgImage, const sensor_msgs::PointCloud2ConstPtr& msgPointCloud);

    ORB_SLAM3::System* mpSLAM;
    double lastTimestamp;

    Eigen::Vector3f lastCameraPos;
    Eigen::Vector3f lastObjectPos;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBL");
    ros::start();
    ros::NodeHandle nh;

    if (argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 RGBL_SLAM path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }

    // 初始化 SLAM 系统
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBL, true);

    ImageGrabber igb(&SLAM);

    // 使用 message_filters 同步图像和点云话题
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/usb_cam/image", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, "/lidar/points", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, cloud_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabImagePointCloud, &igb, _1, _2));

    ros::spin();

    // 关闭 SLAM 系统
    SLAM.Shutdown();

    // 保存相机轨迹
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImagePointCloud(const sensor_msgs::ImageConstPtr& msgImage, const sensor_msgs::PointCloud2ConstPtr& msgPointCloud)
{
    ros::NodeHandle nh;
    ros::Publisher objectRelativePosPub = nh.advertise<geometry_msgs::Point>("object_relative_position", 10);
    ros::Publisher objectWorldPosPub = nh.advertise<geometry_msgs::Point>("object_world_position", 10);
    ros::Publisher cameraWorldPosPub = nh.advertise<geometry_msgs::Point>("camera_world_position", 10);

    ros::Publisher objectRelativeVelPub = nh.advertise<geometry_msgs::Vector3>("object_relative_velocity", 10);
    ros::Publisher objectWorldVelPub = nh.advertise<geometry_msgs::Vector3>("object_world_velocity", 10);
    ros::Publisher cameraWorldVelPub = nh.advertise<geometry_msgs::Vector3>("camera_world_velocity", 10);

    ros::Publisher objectRelativeAccPub = nh.advertise<geometry_msgs::Vector3>("object_relative_acceleration", 10);
    ros::Publisher objectWorldAccPub = nh.advertise<geometry_msgs::Vector3>("object_world_acceleration", 10);
    ros::Publisher cameraWorldAccPub = nh.advertise<geometry_msgs::Vector3>("camera_world_acceleration", 10);

    // 将ROS图像消息转换为cv::Mat格式
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msgImage);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat imRGB = cv_ptr->image;

    // 将ROS点云消息转换为PCL格式的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msgPointCloud, *pcl_cloud);

    // 创建一个深度图
    cv::Mat depth_map(imRGB.rows, imRGB.cols, CV_32F, cv::Scalar(-1));

    // 遍历点云，填充深度图
    for (const auto& point : pcl_cloud->points)
    {
        if (point.z > 0)
        {
            int u = static_cast<int>(point.x * 469.8 / point.z + 320);  // 假设相机内参 fx = 469.8, cx = 320
            int v = static_cast<int>(point.y * 469.8 / point.z + 240);  // 假设相机内参 fy = 469.8, cy = 240

            if (u >= 0 && u < depth_map.cols && v >= 0 && v < depth_map.rows)
            {
                depth_map.at<float>(v, u) = point.z;
            }
        }
    }

    // 获取当前时间戳
    double timestamp = msgImage->header.stamp.toSec();
    
    // 执行SLAM跟踪，获取相机的位姿 Sophus::SE3f 对象
    Sophus::SE3f Tcw_SE3 = mpSLAM->TrackRGBL(imRGB, depth_map, timestamp);
    
    Eigen::Vector3f cameraWorldPos = Tcw_SE3.translation();

    geometry_msgs::Point camPosMsg;
    camPosMsg.x = cameraWorldPos[0];
    camPosMsg.y = cameraWorldPos[1];
    camPosMsg.z = cameraWorldPos[2];
    cameraWorldPosPub.publish(camPosMsg);

    if (lastTimestamp > 0)
    {
        double deltaTime = timestamp - lastTimestamp;

        Eigen::Vector3f cameraVelocity = (cameraWorldPos - lastCameraPos) / deltaTime;

        geometry_msgs::Vector3 camVelMsg;
        camVelMsg.x = cameraVelocity[0];
        camVelMsg.y = cameraVelocity[1];
        camVelMsg.z = cameraVelocity[2];
        cameraWorldVelPub.publish(camVelMsg);

        Eigen::Vector3f cameraAcceleration = (cameraVelocity - (lastCameraPos - cameraWorldPos) / deltaTime) / deltaTime;

        geometry_msgs::Vector3 camAccMsg;
        camAccMsg.x = cameraAcceleration[0];
        camAccMsg.y = cameraAcceleration[1];
        camAccMsg.z = cameraAcceleration[2];
        cameraWorldAccPub.publish(camAccMsg);
    }

    // 使用全局变量 gDynamicArea 获取检测到的物体位置框
    for (const auto& object : gDynamicArea)
    {
        // 获取物体的中心点
        cv::Point2f objectCenter(object.x + object.width / 2, object.y + object.height / 2);

        // 从深度图中获取物体的深度信息
        float depth = depth_map.at<float>(objectCenter.y, objectCenter.x);

        if (depth < 0)
        {
            std::cout << "No valid depth found for object at " << objectCenter << std::endl;
            continue;
        }

        // 将物体的2D位置转换为3D位置
        Eigen::Vector3f objectPosCam;
        objectPosCam[0] = (objectCenter.x - 320) * depth / 469.8;  // 假设相机内参 fx = 469.8, cx = 320
        objectPosCam[1] = (objectCenter.y - 240) * depth / 469.8;  // 假设相机内参 fy = 469.8, cy = 240
        objectPosCam[2] = depth;

        // 发布物体相对相机的坐标
        geometry_msgs::Point relativePosMsg;
        relativePosMsg.x = objectPosCam[0];
        relativePosMsg.y = objectPosCam[1];
        relativePosMsg.z = objectPosCam[2];
        objectRelativePosPub.publish(relativePosMsg);

        // 将物体的相机坐标系位置转换到世界坐标系
        Eigen::Vector3f objectPosWorld = Tcw_SE3.inverse() * objectPosCam;

        // 发布物体的世界坐标
        geometry_msgs::Point worldPosMsg;
        worldPosMsg.x = objectPosWorld[0];
        worldPosMsg.y = objectPosWorld[1];
        worldPosMsg.z = objectPosWorld[2];
        objectWorldPosPub.publish(worldPosMsg);

        if (lastTimestamp > 0)
        {
            double deltaTime = timestamp - lastTimestamp;

            // 计算物体的相对速度 (相对于相机)
            Eigen::Vector3f objectRelativeVelocity = (objectPosCam - lastObjectPos) / deltaTime;
            geometry_msgs::Vector3 relativeVelMsg;
            relativeVelMsg.x = objectRelativeVelocity[0];
            relativeVelMsg.y = objectRelativeVelocity[1];
            relativeVelMsg.z = objectRelativeVelocity[2];
            objectRelativeVelPub.publish(relativeVelMsg);

            // 计算物体的相对加速度 (相对于相机)
            Eigen::Vector3f objectRelativeAcceleration = (objectRelativeVelocity - (lastObjectPos - objectPosCam) / deltaTime) / deltaTime;
            geometry_msgs::Vector3 relativeAccMsg;
            relativeAccMsg.x = objectRelativeAcceleration[0];
            relativeAccMsg.y = objectRelativeAcceleration[1];
            relativeAccMsg.z = objectRelativeAcceleration[2];
            objectRelativeAccPub.publish(relativeAccMsg);

            // 计算物体的绝对速度 (相对于世界坐标系)
            Eigen::Vector3f objectWorldVelocity = (objectPosWorld - Tcw_SE3.inverse() * lastObjectPos) / deltaTime;
            geometry_msgs::Vector3 worldVelMsg;
            worldVelMsg.x = objectWorldVelocity[0];
            worldVelMsg.y = objectWorldVelocity[1];
            worldVelMsg.z = objectWorldVelocity[2];
            objectWorldVelPub.publish(worldVelMsg);

            // 计算物体的绝对加速度 (相对于世界坐标系)
            Eigen::Vector3f objectWorldAcceleration = (objectWorldVelocity - (Tcw_SE3.inverse() * lastObjectPos - objectPosWorld) / deltaTime) / deltaTime;
            geometry_msgs::Vector3 worldAccMsg;
            worldAccMsg.x = objectWorldAcceleration[0];
            worldAccMsg.y = objectWorldAcceleration[1];
            worldAccMsg.z = objectWorldAcceleration[2];
            objectWorldAccPub.publish(worldAccMsg);
        }

        // 更新上一次的物体和相机位置
        lastObjectPos = objectPosCam;
    }

    lastCameraPos = cameraWorldPos;
    lastTimestamp = timestamp;
}