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

void create_visualization_video(const std::vector<cv::Mat>& frames, const std::string& output_path) {
    if (frames.empty()) {
        std::cerr << "没有帧可以用于创建视频" << std::endl;
        return;
    }
    
    int frame_width = frames[0].cols;
    int frame_height = frames[0].rows;
    double fps = 1;  // 进一步降低FPS到2，视频会更长
    
    cv::VideoWriter video_writer(output_path, 
                               cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                               fps, cv::Size(frame_width, frame_height));
    
    if (!video_writer.isOpened()) {
        std::cerr << "无法创建视频文件" << std::endl;
        return;
    }
    
    for (const auto& frame : frames) {
        cv::Mat resized_frame;
        if (frame.cols != frame_width || frame.rows != frame_height) {
            cv::resize(frame, resized_frame, cv::Size(frame_width, frame_height));
            video_writer.write(resized_frame);
        } else {
            video_writer.write(frame);
        }
    }
    
    video_writer.release();
    std::cout << "视频已保存到 " << output_path << " (总共 " << frames.size() << " 帧，FPS=" << fps << "，预计时长=" << (frames.size()/fps) << "秒)" << std::endl;
}

std::vector<cv::Mat> frames_vertical;
std::vector<cv::Mat> frames_horizontal;
std::vector<cv::Mat> frames_combined;
std::vector<cv::Mat> frames_optimized;
std::vector<cv::Mat> frames_forward;
std::vector<cv::Mat> frames_enlarge;
std::vector<cv::Mat> frames_enhance;
std::vector<cv::Mat> frames_energy_map; 

SeamCarver* current_carver_ptr = nullptr;

cv::Mat visualize_seam(const cv::Mat& image, const std::vector<int>& seam) {
    if (seam.empty()) return image.clone();
    
    cv::Mat result = image.clone();
    for (int y = 0; y < result.rows && y < (int)seam.size(); ++y) {
        int x = seam[y];
        if (x >= 0 && x < result.cols) {
            result.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); 
            if (x > 0)
                result.at<cv::Vec3b>(y, x-1) = cv::Vec3b(0, 0, 255);
            if (x < result.cols - 1)
                result.at<cv::Vec3b>(y, x+1) = cv::Vec3b(0, 0, 255);
        }
    }
    return result;
}

void save_frame_callback_vertical(const cv::Mat& current_image, int current, int total, const std::vector<int>& last_seam) {
    update_progress(current_image, current, total);
    
    if (current == 1 || current == total || current % 1 == 0) {
        cv::Mat frame_with_size = current_image.clone();
        
        if (!last_seam.empty()) {
            frame_with_size = visualize_seam(frame_with_size, last_seam);
        }
        
        std::string size_text = std::to_string(frame_with_size.cols) + "x" + 
                             std::to_string(frame_with_size.rows);
        cv::putText(frame_with_size, size_text, cv::Point(10, 30), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        frames_vertical.push_back(frame_with_size);
        
        if (last_seam.size() > 0 && current_carver_ptr != nullptr) {
            cv::Mat energy_map = current_carver_ptr->get_energy_map_heatmap();
            
            for (int y = 0; y < energy_map.rows && y < (int)last_seam.size(); ++y) {
                int x = last_seam[y];
                if (x >= 0 && x < energy_map.cols) {
                    energy_map.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255); 
                    if (x > 0)
                        energy_map.at<cv::Vec3b>(y, x-1) = cv::Vec3b(255, 255, 255);
                    if (x < energy_map.cols - 1)
                        energy_map.at<cv::Vec3b>(y, x+1) = cv::Vec3b(255, 255, 255);
                }
            }
            
            cv::putText(energy_map, "Energy Map " + size_text, cv::Point(10, 30), 
                      cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            
            frames_energy_map.push_back(energy_map);
        }
    }
}

void save_frame_callback_horizontal(const cv::Mat& current_image, int current, int total, const std::vector<int>& last_seam) {
    update_progress(current_image, current, total);
    
    if (current == 1 || current == total || current % 2 == 0) {
        cv::Mat frame_with_size = current_image.clone();
        
        if (!last_seam.empty()) {
            std::vector<int> horizontal_seam(current_image.cols);
            for (int x = 0; x < std::min(current_image.cols, (int)last_seam.size()); ++x) {
                horizontal_seam[x] = last_seam[x];
            }
            
            for (int x = 0; x < frame_with_size.cols && x < (int)horizontal_seam.size(); ++x) {
                int y = horizontal_seam[x];
                if (y >= 0 && y < frame_with_size.rows) {
                    frame_with_size.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); 
                    
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

cv::Mat create_algorithm_comparison(const cv::Mat& original_img, const std::string& output_path) {
    int target_width = original_img.cols * 0.7;
    int target_height = original_img.rows;
    
    SeamCarver carver_basic(original_img.clone(), Algorithm::BASIC);
    SeamCarver carver_optimized(original_img.clone(), Algorithm::OPTIMIZED);
    SeamCarver carver_forward(original_img.clone(), Algorithm::FORWARD_ENERGY);
    
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
    
    cv::Mat basic_result = carver_basic.get_current_image();
    cv::Mat optimized_result = carver_optimized.get_current_image();
    cv::Mat forward_result = carver_forward.get_current_image();
    
    int padding = 20;
    int label_height = 40;
    int total_width = original_img.cols + padding + target_width * 3 + padding * 3;
    int total_height = original_img.rows + label_height + padding * 2;
    
    cv::Mat comparison(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    original_img.copyTo(comparison(cv::Rect(padding, padding + label_height, original_img.cols, original_img.rows)));
    
    basic_result.copyTo(comparison(cv::Rect(padding * 2 + original_img.cols, padding + label_height, target_width, target_height)));
    
    optimized_result.copyTo(comparison(cv::Rect(padding * 3 + original_img.cols + target_width, padding + label_height, target_width, target_height)));
    
    forward_result.copyTo(comparison(cv::Rect(padding * 4 + original_img.cols + target_width * 2, padding + label_height, target_width, target_height)));
    
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
    
    cv::imwrite(output_path, comparison);
    
    return comparison;
}

int main(int argc, char** argv) {
    int vertical_seams_to_remove = 350;
    int horizontal_seams_to_remove = 350;
    bool preview_seams = true;
    bool generate_energy_video = true;
    
    std::string image_path;
    if (argc > 1) {
        image_path = argv[1];
    } else {
        image_path = "1.jpg"; 
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << image_path << std::endl;
        return -1;
    }
    
    std::string output_dir = "seam_carving_output_" + 
                          std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::filesystem::create_directory(output_dir);
    
    cv::imwrite(output_dir + "/original.jpg", image);
    
    create_algorithm_comparison(image, output_dir + "/algorithm_comparison.jpg");
    
    std::cout << "进行垂直方向Seam Carving..." << std::endl;
    
    SeamCarver carver_for_scaling_v(image.clone(), Algorithm::FORWARD_ENERGY);
    current_carver_ptr = &carver_for_scaling_v; 
    
    frames_vertical.clear(); 
    frames_energy_map.clear(); 
    
    if (preview_seams) {
        carver_for_scaling_v.remove_vertical_seams_forward(vertical_seams_to_remove, save_frame_callback_vertical);
    } else {
        carver_for_scaling_v.remove_vertical_seams_forward(vertical_seams_to_remove, update_progress);
    }
    
    cv::Mat result_v = carver_for_scaling_v.get_current_image();
    cv::imwrite(output_dir + "/result_vertical.jpg", result_v);
    
    if (frames_vertical.size() > 1) {
        create_visualization_video(frames_vertical, output_dir + "/vertical_seam_carving.mp4");
    }
    
    if (generate_energy_video && frames_energy_map.size() > 1) {
        create_visualization_video(frames_energy_map, output_dir + "/energy_map_visualization.mp4");
    }
    
    std::cout << "进行水平方向Seam Carving..." << std::endl;
    
    SeamCarver carver_for_scaling_h(result_v.clone(), Algorithm::OPTIMIZED);
    current_carver_ptr = &carver_for_scaling_h; 
    
    if (preview_seams) {
        carver_for_scaling_h.remove_horizontal_seams_optimized(horizontal_seams_to_remove, save_frame_callback_horizontal);
    } else {
        carver_for_scaling_h.remove_horizontal_seams_optimized(horizontal_seams_to_remove, update_progress);
    }
    
    cv::Mat result_h = carver_for_scaling_h.get_current_image();
    cv::imwrite(output_dir + "/result_combined.jpg", result_h);
    
    if (frames_horizontal.size() > 1) {
        create_visualization_video(frames_horizontal, output_dir + "/horizontal_seam_carving.mp4");
    }
    
    current_carver_ptr = nullptr;
    
    cv::Mat scaled_image;
    cv::resize(image, scaled_image, result_h.size(), 0, 0, cv::INTER_AREA);
    cv::imwrite(output_dir + "/result_traditional_scaling.jpg", scaled_image);
    
    int padding = 20;
    int label_height = 40;
    int total_width = result_h.cols * 2 + padding * 3;
    int total_height = result_h.rows + label_height + padding * 2;
    
    cv::Mat final_comparison(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    result_h.copyTo(final_comparison(cv::Rect(padding, padding + label_height, result_h.cols, result_h.rows)));
    
    scaled_image.copyTo(final_comparison(cv::Rect(padding * 2 + result_h.cols, padding + label_height, scaled_image.cols, scaled_image.rows)));
    
    cv::putText(final_comparison, "Seam Carving", cv::Point(padding, padding + 30), 
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(final_comparison, "Traditional Scaling", 
              cv::Point(padding * 2 + result_h.cols, padding + 30), 
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    
    cv::imwrite(output_dir + "/seam_carving_vs_scaling.jpg", final_comparison);
    
    std::cout << "完成！结果保存在 " << output_dir << " 目录下" << std::endl;
    
    return 0;
} 