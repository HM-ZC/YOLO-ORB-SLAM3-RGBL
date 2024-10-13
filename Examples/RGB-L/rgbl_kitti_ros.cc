#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <opencv2/core/core.hpp>
#include <System.h>  // ORB-SLAM3的头文件

using namespace std;

// 回调函数：同步接收图像和点云数据，处理并传递给SLAM系统
void imageCallback(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::PointCloud2ConstPtr& msgPointCloud, ORB_SLAM3::System* SLAM)
{
    // 将ROS图像消息转换为OpenCV格式
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msgRGB, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat imRGB = cv_ptr->image;

    // 将PointCloud2消息转换为PCL格式点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msgPointCloud, *cloud);

    // 将PCL点云数据转换为cv::Mat，适应SLAM系统的输入
    cv::Mat pcd(cloud->points.size(), 1, CV_32FC3);
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        pcd.at<cv::Vec3f>(i)[0] = cloud->points[i].x;
        pcd.at<cv::Vec3f>(i)[1] = cloud->points[i].y;
        pcd.at<cv::Vec3f>(i)[2] = cloud->points[i].z;
    }

    // 获取当前时间戳
    double tframe = ros::Time::now().toSec();

    // 调用SLAM系统的TrackRGBL函数进行跟踪
    SLAM->TrackRGBL(imRGB, pcd, tframe);
}

int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "rgbl_slam_node");
    ros::NodeHandle nh;

    if (argc != 3)
    {
        ROS_ERROR("Usage: rosrun your_package rgbl_slam_node path_to_vocabulary path_to_settings");
        return -1;
    }

    // 初始化ORB-SLAM3系统
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBL, true);

    // 使用message_filters同步RGB图像和点云数据
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/usb_camera/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub(nh, "/velodyne_points", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> syncPolicy;
    message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10), rgb_sub, pc_sub);
    sync.registerCallback(boost::bind(&imageCallback, _1, _2, &SLAM));

    // 进入ROS事件循环
    ros::spin();

    // 停止SLAM系统的所有线程
    SLAM.Shutdown();

    return 0;
}

