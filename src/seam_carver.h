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
    
    cv::Mat get_current_image() const;
    
    cv::Mat get_energy_map() const;
    cv::Mat get_energy_map_heatmap() const;
    
    void remove_vertical_seam(const std::vector<int>& seam);
    
    void remove_vertical_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    void remove_vertical_seams_optimized(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    void remove_vertical_seams_forward(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    void remove_horizontal_seam(const std::vector<int>& seam);
    
    void remove_horizontal_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    void remove_horizontal_seams_optimized(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    cv::Mat show_next_vertical_seam();
    
    cv::Mat show_multiple_vertical_seams(int num_seams);
    
    cv::Mat show_next_horizontal_seam();
    
    cv::Mat show_multiple_horizontal_seams(int num_seams);
    
    void insert_vertical_seam(const std::vector<int>& seam);
    
    void insert_vertical_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    void insert_horizontal_seam(const std::vector<int>& seam);
    
    void insert_horizontal_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    cv::Mat visualize_transport_map(int target_width, int target_height);
    
    void resize_optimal(int target_width, int target_height, 
                       std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback = nullptr);
    
    cv::Mat enhance_subject(int scale_percentage, const std::function<void(const cv::Mat&, int, int, const std::vector<int>&)>& update_callback = nullptr);
    
    std::vector<int> find_vertical_seam() const;
    
    std::vector<int> find_vertical_seam_optimized() const;
    
    std::vector<int> find_vertical_seam_forward() const;
    
    std::vector<int> find_horizontal_seam() const;
    
    std::vector<std::vector<int>> find_multiple_vertical_seams(int num_seams) const;
    
    std::vector<std::vector<int>> find_multiple_horizontal_seams(int num_seams) const;
    
    cv::Mat apply_heatmap_colormap(const cv::Mat& energy) const;
    
    cv::Mat transpose_image(const cv::Mat& img) const;
    
private:
    cv::Mat current_image;
    cv::Mat energy_map;
    Algorithm algorithm;
    
    float cost_for_removal(const cv::Vec3b& pixel1, const cv::Vec3b& pixel2) const;
    
    void calculate_energy_map();
    
    cv::Mat calculate_forward_energy_map() const;
    
    std::pair<cv::Mat, cv::Mat> compute_transport_map(int target_width, int target_height) const;
};

#endif 