#ifndef SEAM_CARVER_H
#define SEAM_CARVER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>

enum class Algorithm {
    BASIC,
    OPTIMIZED,
    FORWARD_ENERGY
};

enum class ResizeMethod {
    STANDARD,
    OPTIMAL,
    TRADITIONAL
};

class SeamCarver {
public:
    SeamCarver(const cv::Mat& image, Algorithm algo = Algorithm::BASIC);
    
    // 获取当前图像
    cv::Mat get_current_image() const;
    
    // 计算能量图并获取可视化
    cv::Mat get_energy_map() const;
    cv::Mat get_energy_map_heatmap() const;
    
    // 移除垂直seam
    void remove_vertical_seam(const std::vector<int>& seam);
    
    // 移除多个垂直seam
    void remove_vertical_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    // 使用优化算法移除垂直seam
    void remove_vertical_seams_optimized(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    // 使用前向能量算法移除垂直seam
    void remove_vertical_seams_forward(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    // 移除水平seam
    void remove_horizontal_seam(const std::vector<int>& seam);
    
    // 移除多个水平seam
    void remove_horizontal_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    // 优化版：移除水平seam
    void remove_horizontal_seams_optimized(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    // 显示下一个要移除的垂直seam
    cv::Mat show_next_vertical_seam();
    
    // 显示多个垂直seam
    cv::Mat show_multiple_vertical_seams(int num_seams);
    
    // 显示下一个要移除的水平seam
    cv::Mat show_next_horizontal_seam();
    
    // 显示多个水平seam
    cv::Mat show_multiple_horizontal_seams(int num_seams);
    
    // 插入垂直seam（图像放大）
    void insert_vertical_seam(const std::vector<int>& seam);
    
    // 插入多个垂直seam
    void insert_vertical_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    // 插入水平seam
    void insert_horizontal_seam(const std::vector<int>& seam);
    
    // 插入多个水平seam
    void insert_horizontal_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    // 计算传输图并可视化
    cv::Mat visualize_transport_map(int target_width, int target_height);
    
    // 使用传输图最优顺序调整大小
    void resize_optimal(int target_width, int target_height, 
                       std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    // 主成分增强 - 放大图像然后重新裁剪以突出重要内容
    cv::Mat enhance_subject(int scale_percentage, const std::function<void(const cv::Mat&, int, int, const std::vector<int>&)>& update_callback = nullptr);
    
    // 寻找垂直seam
    std::vector<int> find_vertical_seam() const;
    
    // 寻找垂直seam（优化版）
    std::vector<int> find_vertical_seam_optimized() const;
    
    // 寻找前向能量垂直seam
    std::vector<int> find_vertical_seam_forward() const;
    
    // 寻找水平seam
    std::vector<int> find_horizontal_seam() const;
    
    // 寻找多条垂直seam
    std::vector<std::vector<int>> find_multiple_vertical_seams(int num_seams) const;
    
    // 寻找多条水平seam
    std::vector<std::vector<int>> find_multiple_horizontal_seams(int num_seams) const;
    
    // 应用热力图着色
    cv::Mat apply_heatmap_colormap(const cv::Mat& energy) const;
    
    // 转置图像
    cv::Mat transpose_image(const cv::Mat& img) const;
    
private:
    cv::Mat current_image;
    cv::Mat energy_map;
    Algorithm algorithm;
    
    // 计算移除像素后的能量成本
    float cost_for_removal(const cv::Vec3b& pixel1, const cv::Vec3b& pixel2) const;
    
    // 计算能量图
    void calculate_energy_map();
    
    // 计算前向能量图
    cv::Mat calculate_forward_energy_map() const;
    
    // 计算传输图
    std::pair<cv::Mat, cv::Mat> compute_transport_map(int target_width, int target_height) const;
};

#endif // SEAM_CARVER_H 