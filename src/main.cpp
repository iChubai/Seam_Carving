#include <opencv2/opencv.hpp>
#include "seam_carver.h"
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <functional>

// 进度更新回调函数 - 保存当前帧到文件
void update_progress(const cv::Mat& current_image, int current, int total, const std::vector<int>& last_seam = std::vector<int>()) {
    int progress = static_cast<int>((static_cast<float>(current) / total) * 100);
    std::cout << "\r进度: [";
    int pos = 50 * progress / 100;
    for (int i = 0; i < 50; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << progress << "% (" << current << "/" << total << ")";
    std::cout.flush();
}

// 创建可视化视频
void create_visualization_video(const std::vector<cv::Mat>& frames, const std::string& output_path) {
    if (frames.empty()) {
        std::cerr << "没有帧可以用于创建视频" << std::endl;
        return;
    }
    
    // 设置视频参数
    int frame_width = frames[0].cols;
    int frame_height = frames[0].rows;
    double fps = 1.0; // 每秒帧数
    
    // 创建视频写入器
    cv::VideoWriter video_writer(output_path, 
                               cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                               fps, cv::Size(frame_width, frame_height));
    
    if (!video_writer.isOpened()) {
        std::cerr << "无法创建视频文件" << std::endl;
        return;
    }
    
    // 将每一帧写入视频
    for (const auto& frame : frames) {
        // 确保所有帧的尺寸相同
        cv::Mat resized_frame;
        if (frame.cols != frame_width || frame.rows != frame_height) {
            cv::resize(frame, resized_frame, cv::Size(frame_width, frame_height));
            video_writer.write(resized_frame);
        } else {
            video_writer.write(frame);
        }
    }
    
    // 关闭视频写入器
    video_writer.release();
    std::cout << "视频已保存到 " << output_path << std::endl;
}

// 使用全局变量存储帧
std::vector<cv::Mat> frames_vertical;
std::vector<cv::Mat> frames_horizontal;
std::vector<cv::Mat> frames_combined;
std::vector<cv::Mat> frames_optimized;
std::vector<cv::Mat> frames_forward;
std::vector<cv::Mat> frames_enlarge;
std::vector<cv::Mat> frames_enhance;
std::vector<cv::Mat> frames_energy_map; // 新增：存储能量图帧

// 当前正在使用的SeamCarver实例（用于获取能量图）
SeamCarver* current_carver_ptr = nullptr;

// 在带有seam路径的图像上显示seam线
cv::Mat visualize_seam(const cv::Mat& image, const std::vector<int>& seam) {
    if (seam.empty()) return image.clone();
    
    cv::Mat result = image.clone();
    // 用红色标记seam
    for (int y = 0; y < result.rows && y < (int)seam.size(); ++y) {
        int x = seam[y];
        if (x >= 0 && x < result.cols) {
            result.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // 红色
            
            // 使seam线更粗，便于观察
            if (x > 0)
                result.at<cv::Vec3b>(y, x-1) = cv::Vec3b(0, 0, 255);
            if (x < result.cols - 1)
                result.at<cv::Vec3b>(y, x+1) = cv::Vec3b(0, 0, 255);
        }
    }
    return result;
}

// 带保存帧的回调函数 - 包含seam可视化
void save_frame_callback_vertical(const cv::Mat& current_image, int current, int total, const std::vector<int>& last_seam) {
    update_progress(current_image, current, total);
    
    // 每隔几帧保存一次，以避免生成太多图像
    if (current == 1 || current == total || current % std::max(1, total / 20) == 0) {
        // 创建带尺寸标签的帧
        cv::Mat frame_with_size = current_image.clone();
        
        // 在图像上显示seam线
        if (!last_seam.empty()) {
            frame_with_size = visualize_seam(frame_with_size, last_seam);
        }
        
        std::string size_text = std::to_string(frame_with_size.cols) + "x" + 
                             std::to_string(frame_with_size.rows);
        cv::putText(frame_with_size, size_text, cv::Point(10, 30), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        frames_vertical.push_back(frame_with_size);
        
        // 同时保存当前能量图
        if (last_seam.size() > 0 && current_carver_ptr != nullptr) {
            // 创建能量图的副本
            cv::Mat energy_map = current_carver_ptr->get_energy_map_heatmap();
            
            // 在能量图上标记seam
            for (int y = 0; y < energy_map.rows && y < (int)last_seam.size(); ++y) {
                int x = last_seam[y];
                if (x >= 0 && x < energy_map.cols) {
                    energy_map.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255); // 白色
                    // 使seam线更粗，便于观察
                    if (x > 0)
                        energy_map.at<cv::Vec3b>(y, x-1) = cv::Vec3b(255, 255, 255);
                    if (x < energy_map.cols - 1)
                        energy_map.at<cv::Vec3b>(y, x+1) = cv::Vec3b(255, 255, 255);
                }
            }
            
            // 添加尺寸信息
            cv::putText(energy_map, "Energy Map " + size_text, cv::Point(10, 30), 
                      cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            
            frames_energy_map.push_back(energy_map);
        }
    }
}

void save_frame_callback_horizontal(const cv::Mat& current_image, int current, int total, const std::vector<int>& last_seam) {
    update_progress(current_image, current, total);
    
    if (current == 1 || current == total || current % std::max(1, total / 20) == 0) {
        // 创建带尺寸标签的帧
        cv::Mat frame_with_size = current_image.clone();
        
        // 在图像上显示seam线
        if (!last_seam.empty()) {
            // 对于水平seam，我们需要转换坐标
            std::vector<int> horizontal_seam(current_image.cols);
            for (int x = 0; x < std::min(current_image.cols, (int)last_seam.size()); ++x) {
                horizontal_seam[x] = last_seam[x];
            }
            
            // 在水平方向标记seam
            for (int x = 0; x < frame_with_size.cols && x < (int)horizontal_seam.size(); ++x) {
                int y = horizontal_seam[x];
                if (y >= 0 && y < frame_with_size.rows) {
                    frame_with_size.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // 红色
                    
                    // 使seam线更粗，便于观察
                    if (y > 0)
                        frame_with_size.at<cv::Vec3b>(y-1, x) = cv::Vec3b(0, 0, 255);
                    if (y < frame_with_size.rows - 1)
                        frame_with_size.at<cv::Vec3b>(y+1, x) = cv::Vec3b(0, 0, 255);
                }
            }
        }
        
        std::string size_text = std::to_string(frame_with_size.cols) + "x" + 
                             std::to_string(frame_with_size.rows);
        cv::putText(frame_with_size, size_text, cv::Point(10, 30), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        frames_horizontal.push_back(frame_with_size);
    }
}

// 使用示例: 在一个图像上显示三种算法的对比结果
cv::Mat create_algorithm_comparison(const cv::Mat& original_img, const std::string& output_path) {
    int target_width = original_img.cols * 0.7;
    int target_height = original_img.rows;
    
    // 保存原始图像作为比较
    SeamCarver carver_basic(original_img.clone(), Algorithm::BASIC);
    SeamCarver carver_optimized(original_img.clone(), Algorithm::OPTIMIZED);
    SeamCarver carver_forward(original_img.clone(), Algorithm::FORWARD_ENERGY);
    
    // 开始计时
    auto start_basic = std::chrono::high_resolution_clock::now();
    carver_basic.remove_vertical_seams(original_img.cols - target_width);
    auto end_basic = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_basic = end_basic - start_basic;
    
    auto start_optimized = std::chrono::high_resolution_clock::now();
    carver_optimized.remove_vertical_seams_optimized(original_img.cols - target_width);
    auto end_optimized = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_optimized = end_optimized - start_optimized;
    
    auto start_forward = std::chrono::high_resolution_clock::now();
    carver_forward.remove_vertical_seams_forward(original_img.cols - target_width);
    auto end_forward = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_forward = end_forward - start_forward;
    
    // 获取结果
    cv::Mat basic_result = carver_basic.get_current_image();
    cv::Mat optimized_result = carver_optimized.get_current_image();
    cv::Mat forward_result = carver_forward.get_current_image();
    
    // 创建一个包含所有图像的大图
    int padding = 20;
    int label_height = 40;
    int total_width = original_img.cols + padding + target_width * 3 + padding * 3;
    int total_height = original_img.rows + label_height + padding * 2;
    
    cv::Mat comparison(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // 放置原始图像
    original_img.copyTo(comparison(cv::Rect(padding, padding + label_height, original_img.cols, original_img.rows)));
    
    // 放置基本算法结果
    basic_result.copyTo(comparison(cv::Rect(padding * 2 + original_img.cols, padding + label_height, target_width, target_height)));
    
    // 放置优化算法结果
    optimized_result.copyTo(comparison(cv::Rect(padding * 3 + original_img.cols + target_width, padding + label_height, target_width, target_height)));
    
    // 放置前向能量算法结果
    forward_result.copyTo(comparison(cv::Rect(padding * 4 + original_img.cols + target_width * 2, padding + label_height, target_width, target_height)));
    
    // 添加标签
    cv::putText(comparison, "Original image", cv::Point(padding, padding + 30), 
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(comparison, "Basic algorithm " + std::to_string(int(time_basic.count())) + "ms", 
              cv::Point(padding * 2 + original_img.cols, padding + 30), 
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(comparison, "Optimized algorithm " + std::to_string(int(time_optimized.count())) + "ms", 
              cv::Point(padding * 3 + original_img.cols + target_width, padding + 30), 
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(comparison, "Forward energy algorithm " + std::to_string(int(time_forward.count())) + "ms", 
              cv::Point(padding * 4 + original_img.cols + target_width * 2, padding + 30), 
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    
    // 保存比较图
    cv::imwrite(output_path, comparison);
    
    return comparison;
}

int main(int argc, char** argv) {
    // 配置选项
    int vertical_seams_to_remove = 200;
    int horizontal_seams_to_remove = 100;
    bool preview_seams = true;
    bool generate_energy_video = true; // 新选项：是否生成能量图视频
    
    // 获取图像路径
    std::string image_path;
    if (argc > 1) {
        image_path = argv[1];
    } else {
        image_path = "1.jpg"; // 默认图像
    }
    
    // 读取图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << image_path << std::endl;
        return -1;
    }
    
    // 创建输出目录
    std::string output_dir = "seam_carving_output_" + 
                          std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::filesystem::create_directory(output_dir);
    
    // 保存原始图像
    cv::imwrite(output_dir + "/original.jpg", image);
    
    // 创建算法比较图
    create_algorithm_comparison(image, output_dir + "/algorithm_comparison.jpg");
    
    // 垂直方向Seam Carving
    std::cout << "进行垂直方向Seam Carving..." << std::endl;
    
    SeamCarver carver_for_scaling_v(image.clone(), Algorithm::FORWARD_ENERGY);
    current_carver_ptr = &carver_for_scaling_v; // 设置当前carver
    
    frames_vertical.clear(); // 清除之前的帧
    frames_energy_map.clear(); // 清除之前的能量图帧
    
    if (preview_seams) {
        // 显示和移除20条垂直seam
        carver_for_scaling_v.remove_vertical_seams_forward(vertical_seams_to_remove, save_frame_callback_vertical);
    } else {
        // 不显示seam，只显示进度
        carver_for_scaling_v.remove_vertical_seams_forward(vertical_seams_to_remove, update_progress);
    }
    
    // 保存结果
    cv::Mat result_v = carver_for_scaling_v.get_current_image();
    cv::imwrite(output_dir + "/result_vertical.jpg", result_v);
    
    // 如果有足够的帧，创建视频
    if (frames_vertical.size() > 1) {
        create_visualization_video(frames_vertical, output_dir + "/vertical_seam_carving.mp4");
    }
    
    // 如果开启了能量图视频生成并且有足够的帧，创建能量图视频
    if (generate_energy_video && frames_energy_map.size() > 1) {
        create_visualization_video(frames_energy_map, output_dir + "/energy_map_visualization.mp4");
    }
    
    // 水平方向Seam Carving
    std::cout << "进行水平方向Seam Carving..." << std::endl;
    
    SeamCarver carver_for_scaling_h(result_v.clone(), Algorithm::OPTIMIZED);
    current_carver_ptr = &carver_for_scaling_h; // 设置当前carver
    
    frames_horizontal.clear(); // 清除之前的帧
    
    if (preview_seams) {
        carver_for_scaling_h.remove_horizontal_seams_optimized(horizontal_seams_to_remove, save_frame_callback_horizontal);
    } else {
        carver_for_scaling_h.remove_horizontal_seams_optimized(horizontal_seams_to_remove, update_progress);
    }
    
    // 保存结果
    cv::Mat result_h = carver_for_scaling_h.get_current_image();
    cv::imwrite(output_dir + "/result_combined.jpg", result_h);
    
    // 如果有足够的帧，创建视频
    if (frames_horizontal.size() > 1) {
        create_visualization_video(frames_horizontal, output_dir + "/horizontal_seam_carving.mp4");
    }
    
    // 清除当前carver指针
    current_carver_ptr = nullptr;
    
    // 统一缩放对比
    cv::Mat scaled_image;
    cv::resize(image, scaled_image, result_h.size(), 0, 0, cv::INTER_AREA);
    cv::imwrite(output_dir + "/result_traditional_scaling.jpg", scaled_image);
    
    // 显示比较图
    int padding = 20;
    int label_height = 40;
    int total_width = result_h.cols * 2 + padding * 3;
    int total_height = result_h.rows + label_height + padding * 2;
    
    cv::Mat final_comparison(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // 放置Seam Carving结果
    result_h.copyTo(final_comparison(cv::Rect(padding, padding + label_height, result_h.cols, result_h.rows)));
    
    // 放置传统缩放结果
    scaled_image.copyTo(final_comparison(cv::Rect(padding * 2 + result_h.cols, padding + label_height, scaled_image.cols, scaled_image.rows)));
    
    // 添加标签
    cv::putText(final_comparison, "Seam Carving", cv::Point(padding, padding + 30), 
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(final_comparison, "Traditional Scaling", 
              cv::Point(padding * 2 + result_h.cols, padding + 30), 
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    
    // 保存比较图
    cv::imwrite(output_dir + "/seam_carving_vs_scaling.jpg", final_comparison);
    
    std::cout << "完成！结果保存在 " << output_dir << " 目录下" << std::endl;
    
    return 0;
} 