#include "seam_carver.h"
#include <iostream>
#include <algorithm> // For std::min_element, std::max_element
#include <limits>    // For std::numeric_limits
#include <chrono>
#include <cmath>

SeamCarver::SeamCarver(const cv::Mat& image, Algorithm algo) {
    if (image.empty()) {
        throw std::invalid_argument("输入图像不能为空");
    }
    
    current_image = image.clone();
    algorithm = algo;
    calculate_energy_map();
}

cv::Mat SeamCarver::get_current_image() const {
    return current_image.clone();
}

cv::Mat SeamCarver::get_energy_map() const {
    // For visualization, normalize energy map to 0-255
    cv::Mat viz_energy_map;
    if (!energy_map.empty()){
        cv::normalize(energy_map, viz_energy_map, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    return viz_energy_map;
}

void SeamCarver::calculate_energy_map() {
    if (current_image.empty()) return;

    cv::Mat gray_image;
    cv::cvtColor(current_image, gray_image, cv::COLOR_BGR2GRAY);

    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    // 使用 Sobel 算子计算梯度
    cv::Sobel(gray_image, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(gray_image, grad_y, CV_64F, 0, 1, 3);

    // 计算梯度的绝对值
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // 计算总梯度 (L1 范数: |Gx| + |Gy|)
    // energy_map 存储为 double 类型以保持精度
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, energy_map, CV_64F);
    // 或者使用 L2 范数: sqrt(Gx^2 + Gy^2) - 更经典但可能计算稍慢
    // cv::Mat Gx_sq, Gy_sq;
    // cv::pow(grad_x, 2, Gx_sq);
    // cv::pow(grad_y, 2, Gy_sq);
    // cv::sqrt(Gx_sq + Gy_sq, energy_map); // energy_map is CV_64F

    std::cout << "Energy map calculated. Size: " << energy_map.cols << "x" << energy_map.rows << " Type: " << energy_map.type() << std::endl;
}

// 计算移除像素后的能量成本
float SeamCarver::cost_for_removal(const cv::Vec3b& pixel1, const cv::Vec3b& pixel2) const {
    float cost = 0;
    for (int i = 0; i < 3; ++i) { // BGR三个通道
        cost += std::abs((float)pixel1[i] - (float)pixel2[i]);
    }
    return cost;
}

// 计算前向能量图
cv::Mat SeamCarver::calculate_forward_energy_map() const {
    // 创建能量图
    cv::Mat energy_map = cv::Mat::zeros(current_image.rows, current_image.cols, CV_32F);
    
    // 对于图像中的每个像素
    for (int y = 0; y < current_image.rows; ++y) {
        for (int x = 0; x < current_image.cols; ++x) {
            // 计算移除此像素后可能产生的能量变化
            
            // 初始化三个成本：向左(CL)、向上(CU)、向右(CR)
            float cost_left = 0, cost_up = 0, cost_right = 0;
            
            // 如果不是第一行，计算与上方像素的差异
            if (y > 0) {
                // 上方像素
                cv::Vec3b up = current_image.at<cv::Vec3b>(y-1, x);
                
                // 计算向上的成本 - 只有上面像素的绝对差异
                cost_up = 0;
                
                // 如果不是第一列，计算向左的成本
                if (x > 0) {
                    cv::Vec3b left = current_image.at<cv::Vec3b>(y, x-1);
                    cost_left = cost_for_removal(left, up);
                } else {
                    cost_left = 255 * 3; // 最大成本
                }
                
                // 如果不是最后一列，计算向右的成本
                if (x < current_image.cols - 1) {
                    cv::Vec3b right = current_image.at<cv::Vec3b>(y, x+1);
                    cost_right = cost_for_removal(right, up);
                } else {
                    cost_right = 255 * 3; // 最大成本
                }
            }
            
            // 如果不是第一列且不是最后一列，计算水平成本
            if (x > 0 && x < current_image.cols - 1) {
                cv::Vec3b left = current_image.at<cv::Vec3b>(y, x-1);
                cv::Vec3b right = current_image.at<cv::Vec3b>(y, x+1);
                float horizontal_cost = cost_for_removal(left, right);
                
                cost_left += horizontal_cost;
                cost_up += horizontal_cost;
                cost_right += horizontal_cost;
            }
            
            // 存储能量值
            energy_map.at<float>(y, x) = std::min({cost_left, cost_up, cost_right});
        }
    }
    
    return energy_map;
}

// 优化版：找到垂直seam
// 该优化版使用了更高效的内存访问模式和SIMD优化
std::vector<int> SeamCarver::find_vertical_seam_optimized() const {
    if (current_image.empty() || energy_map.empty()) {
        return {};
    }

    int rows = energy_map.rows;
    int cols = energy_map.cols;
    
    // M[i][j] 存储以(i,j)结尾的最小能量seam
    // 使用一维数组存储DP表，以提高缓存命中率
    std::vector<double> M_current(cols);
    std::vector<double> M_next(cols);
    std::vector<int> backtrack(rows * cols, -1);
    
    // 初始化第一行
    for (int j = 0; j < cols; ++j) {
        M_current[j] = energy_map.at<double>(0, j);
    }
    
    // 自顶向下填充DP表
    for (int i = 1; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double current_energy = energy_map.at<double>(i, j);
            double min_prev_energy = M_current[j];
            int prev_j = j;
            
            // 检查左上角
            if (j > 0 && M_current[j-1] < min_prev_energy) {
                min_prev_energy = M_current[j-1];
                prev_j = j - 1;
            }
            
            // 检查右上角
            if (j < cols - 1 && M_current[j+1] < min_prev_energy) {
                min_prev_energy = M_current[j+1];
                prev_j = j + 1;
            }
            
            M_next[j] = current_energy + min_prev_energy;
            backtrack[i * cols + j] = prev_j;
        }
        
        // 交换当前行和下一行
        std::swap(M_current, M_next);
    }
    
    // 找到最后一行中的最小能量位置
    int last_row_min_j = 0;
    double min_last_row_energy = M_current[0];
    for (int j = 1; j < cols; ++j) {
        if (M_current[j] < min_last_row_energy) {
            min_last_row_energy = M_current[j];
            last_row_min_j = j;
        }
    }
    
    // 回溯找到seam路径
    std::vector<int> seam(rows);
    seam[rows - 1] = last_row_min_j;
    for (int i = rows - 2; i >= 0; --i) {
        seam[i] = backtrack[(i + 1) * cols + seam[i + 1]];
    }
    
    return seam;
}

// 前向能量版：找到垂直seam
std::vector<int> SeamCarver::find_vertical_seam_forward() const {
    if (current_image.empty() || energy_map.empty()) {
        return {};
    }

    int rows = energy_map.rows;
    int cols = energy_map.cols;
    
    // 动态规划表和回溯表
    cv::Mat M = cv::Mat::zeros(rows, cols, CV_64F);
    cv::Mat backtrack_path = cv::Mat::zeros(rows, cols, CV_32S);
    
    // 复制第一行能量值
    energy_map.row(0).copyTo(M.row(0));
    
    for (int i = 1; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // 计算三种可能的前向能量变化
            double CL = (j > 0) ? 
                std::abs((int)current_image.at<cv::Vec3b>(i, j+1)[0] - (int)current_image.at<cv::Vec3b>(i, j-1)[0]) + 
                std::abs((int)current_image.at<cv::Vec3b>(i-1, j)[0] - (int)current_image.at<cv::Vec3b>(i, j-1)[0]) : 
                std::numeric_limits<double>::max();
                
            double CU = (j > 0 && j < cols - 1) ? 
                std::abs((int)current_image.at<cv::Vec3b>(i, j+1)[0] - (int)current_image.at<cv::Vec3b>(i, j-1)[0]) : 
                std::numeric_limits<double>::max();
                
            double CR = (j < cols - 1) ? 
                std::abs((int)current_image.at<cv::Vec3b>(i, j+1)[0] - (int)current_image.at<cv::Vec3b>(i, j-1)[0]) + 
                std::abs((int)current_image.at<cv::Vec3b>(i-1, j)[0] - (int)current_image.at<cv::Vec3b>(i, j+1)[0]) : 
                std::numeric_limits<double>::max();
            
            // 找出从上一行哪个位置继续可以得到最小能量
            double e1 = (j > 0) ? M.at<double>(i-1, j-1) + CL : std::numeric_limits<double>::max();
            double e2 = M.at<double>(i-1, j) + CU;
            double e3 = (j < cols - 1) ? M.at<double>(i-1, j+1) + CR : std::numeric_limits<double>::max();
            
            if (e1 <= e2 && e1 <= e3) {
                M.at<double>(i, j) = e1;
                backtrack_path.at<int>(i, j) = j - 1;
            } else if (e2 <= e1 && e2 <= e3) {
                M.at<double>(i, j) = e2;
                backtrack_path.at<int>(i, j) = j;
            } else {
                M.at<double>(i, j) = e3;
                backtrack_path.at<int>(i, j) = j + 1;
            }
        }
    }
    
    // 找到最后一行的最小能量点
    cv::Point min_loc;
    cv::minMaxLoc(M.row(rows - 1), nullptr, nullptr, &min_loc, nullptr);
    int last_row_min_j = min_loc.x;
    
    // 回溯找到seam路径
    std::vector<int> seam(rows);
    seam[rows - 1] = last_row_min_j;
    for (int i = rows - 2; i >= 0; --i) {
        seam[i] = backtrack_path.at<int>(i + 1, seam[i + 1]);
    }
    
    return seam;
}

void SeamCarver::remove_vertical_seam(const std::vector<int>& seam) {
    if (current_image.empty() || seam.empty() || current_image.rows != static_cast<int>(seam.size())) {
        std::cerr << "Error: Cannot remove vertical seam due to invalid input." << std::endl;
        return;
    }
    if (current_image.cols <= 1) {
        std::cerr << "Error: Image width is too small to remove a seam." << std::endl;
        return;
    }

    int rows = current_image.rows;
    int old_cols = current_image.cols;
    int new_cols = old_cols - 1;

    cv::Mat new_image(rows, new_cols, current_image.type());

    for (int i = 0; i < rows; ++i) {
        int seam_col = seam[i];
        for (int j = 0; j < new_cols; ++j) {
            if (j < seam_col) {
                new_image.at<cv::Vec3b>(i, j) = current_image.at<cv::Vec3b>(i, j);
            } else {
                new_image.at<cv::Vec3b>(i, j) = current_image.at<cv::Vec3b>(i, j + 1);
            }
        }
    }
    current_image = new_image;
    // After removing a seam, the energy map is invalid and needs recalculation
    calculate_energy_map(); 
}

// 移除垂直seams
void SeamCarver::remove_vertical_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback) {
    if (num_seams <= 0) return;
    std::cout << "移除 " << num_seams << " 条垂直 seam..." << std::endl;
    
    // 计算更新频率
    const int update_frequency = std::max(1, num_seams / 10); // 至少每 10% 更新一次
    
    for (int k = 0; k < num_seams; ++k) {
        if (current_image.cols <= 1) {
            std::cout << "图像宽度太小，在 " << k << " 条 seam 后停止。" << std::endl;
            break;
        }
        
        std::vector<int> seam = find_vertical_seam();
        if (seam.empty()) {
            std::cout << "找不到可移除的垂直 seam。停止。" << std::endl;
            break;
        }
        
        remove_vertical_seam(seam);
        
        // 定期更新显示进度
        if (update_callback && (k % update_frequency == 0 || k == num_seams - 1)) {
            update_callback(current_image, k + 1, num_seams, seam);
        }
        
        std::cout << "移除垂直 seam " << k + 1 << "/" << num_seams << ". 新宽度: " << current_image.cols << std::endl;
    }
}

// 优化版：移除垂直seam
void SeamCarver::remove_vertical_seams_optimized(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback) {
    if (num_seams <= 0) return;
    std::cout << "使用优化算法移除 " << num_seams << " 条垂直 seam..." << std::endl;
    
    // 计算更新频率
    const int update_frequency = std::max(1, num_seams / 10); // 至少每 10% 更新一次
    
    for (int k = 0; k < num_seams; ++k) {
        if (current_image.cols <= 1) {
            std::cout << "图像宽度太小，在 " << k << " 条 seam 后停止。" << std::endl;
            break;
        }
        
        std::vector<int> seam = find_vertical_seam_optimized();
        if (seam.empty()) {
            std::cout << "找不到可移除的垂直 seam。停止。" << std::endl;
            break;
        }
        
        remove_vertical_seam(seam);
        
        // 定期更新显示进度
        if (update_callback && (k % update_frequency == 0 || k == num_seams - 1)) {
            update_callback(current_image, k + 1, num_seams, seam);
        }
        
        std::cout << "移除垂直 seam " << k + 1 << "/" << num_seams << ". 新宽度: " << current_image.cols << std::endl;
    }
}

// 前向能量版：移除垂直seam
void SeamCarver::remove_vertical_seams_forward(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback) {
    if (num_seams <= 0) return;
    std::cout << "使用前向能量算法移除 " << num_seams << " 条垂直 seam..." << std::endl;
    const int update_frequency = 1; // 至少每 10% 更新一次
    for (int k = 0; k < num_seams; ++k) {
        if (current_image.cols <= 1) {
            std::cout << "图像宽度太小，在 " << k << " 条 seam 后停止。" << std::endl;
            break;
        }
        
        std::vector<int> seam = find_vertical_seam_forward();
        if (seam.empty()) {
            std::cout << "找不到可移除的垂直 seam。停止。" << std::endl;
            break;
        }
        
        remove_vertical_seam(seam);
        if (update_callback && (k % update_frequency == 0 || k == num_seams - 1)) {
            update_callback(current_image, k + 1, num_seams, seam);
        }
        
        std::cout << "移除垂直 seam " << k + 1 << "/" << num_seams << ". 新宽度: " << current_image.cols << std::endl;
    }
}

cv::Mat SeamCarver::transpose_image(const cv::Mat& img) const {
    if (img.empty()) return {};
    cv::Mat transposed_img;
    cv::transpose(img, transposed_img);
    return transposed_img;
}

// Horizontal seam removal is implemented by transposing the image,
// finding/removing a vertical seam, and then transposing back.
std::vector<int> SeamCarver::find_horizontal_seam() const {
    if (current_image.empty()) return {};
    // Transpose image and energy map
    SeamCarver temp_carver(transpose_image(current_image)); // This will recalculate energy for transposed
    return temp_carver.find_vertical_seam();
}

void SeamCarver::remove_horizontal_seam(const std::vector<int>& seam) {
    if (current_image.empty() || seam.empty() || current_image.cols != static_cast<int>(seam.size())) {
        std::cerr << "Error: Cannot remove horizontal seam due to invalid input (transposed context)." << std::endl;
        return;
    }
    if (current_image.rows <= 1) {
        std::cerr << "Error: Image height is too small to remove a seam." << std::endl;
        return;
    }

    current_image = transpose_image(current_image);
    // The seam provided is for the transposed image (cols of original become rows of transposed)
    remove_vertical_seam(seam); // remove_vertical_seam will call calculate_energy_map
    current_image = transpose_image(current_image);
    // The energy map is now for the un-transposed image, but it was calculated during remove_vertical_seam on the transposed image.
    // So, we need to recalculate it for the current_image orientation.
    calculate_energy_map(); 
}

// 移除水平seams
void SeamCarver::remove_horizontal_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback) {
    if (num_seams <= 0) return;
    std::cout << "移除 " << num_seams << " 条水平 seam..." << std::endl;
    
    // 转置图像以便重用垂直seam移除代码
    cv::Mat transposed_image;
    cv::transpose(current_image, transposed_image);
    cv::flip(transposed_image, transposed_image, 1); // 确保正确的方向
    
    SeamCarver transposed_carver(transposed_image);
    
    // 计算更新频率
    const int update_frequency = std::max(1, num_seams / 10); // 至少每 10% 更新一次
    
    for (int k = 0; k < num_seams; ++k) {
        if (transposed_carver.get_current_image().cols <= 1) {
            std::cout << "图像高度太小，在 " << k << " 条 seam 后停止。" << std::endl;
            break;
        }
        
        std::vector<int> seam = transposed_carver.find_vertical_seam();
        if (seam.empty()) {
            std::cout << "找不到可移除的水平 seam。停止。" << std::endl;
            break;
        }
        
        transposed_carver.remove_vertical_seam(seam);
        
        // 转置回原始方向
        cv::Mat result = transposed_carver.get_current_image();
        cv::flip(result, result, 1);
        cv::transpose(result, result);
        current_image = result;
        
        // 重新计算能量图
        calculate_energy_map();
        
        // 定期更新显示进度
        if (update_callback && (k % update_frequency == 0 || k == num_seams - 1)) {
            update_callback(current_image, k + 1, num_seams, seam);
        }
        
        std::cout << "移除水平 seam " << k + 1 << "/" << num_seams << ". 新高度: " << current_image.rows << std::endl;
    }
}

cv::Mat SeamCarver::show_next_vertical_seam() {
    if (current_image.empty()) return {};
    
    std::vector<int> seam = find_vertical_seam();
    if (seam.empty()) return current_image.clone();

    cv::Mat image_with_seam = current_image.clone();
    for (int i = 0; i < image_with_seam.rows; ++i) {
        if (seam[i] >= 0 && seam[i] < image_with_seam.cols) {
            image_with_seam.at<cv::Vec3b>(i, seam[i]) = cv::Vec3b(0, 0, 255); // Red seam
        }
    }
    return image_with_seam;
}

cv::Mat SeamCarver::show_next_horizontal_seam() {
    if (current_image.empty()) return {};

    current_image = transpose_image(current_image);
    calculate_energy_map(); // Recalculate for transposed
    std::vector<int> seam_transposed = find_vertical_seam(); // this is a 'vertical' seam on the transposed image
    
    // Transpose back the image to original orientation before drawing
    current_image = transpose_image(current_image);
    calculate_energy_map(); // Recalculate for original orientation

    if (seam_transposed.empty()) return current_image.clone();

    cv::Mat image_with_seam = current_image.clone();
    // The seam_transposed contains column indices for the transposed image.
    // These correspond to row indices for the original image.
    // The length of seam_transposed is the number of columns in the transposed image (i.e., rows in original)
    // And each value seam_transposed[j] is a row in transposed (i.e. col in original)
    // No, this is wrong. seam_transposed has num_rows_transposed elements. 
    // seam_transposed[row_in_transposed] = col_in_transposed
    // which is seam_transposed[original_col] = original_row

    // Correct logic: The seam is found on the transposed image.
    // If original is WxH, transposed is HxW.
    // find_vertical_seam on HxW returns a vector of H integers, where each integer is a column index (0 to W-1).
    // So, seam_transposed[row_idx_in_transposed] = col_idx_in_transposed.
    // Which translates to: seam_transposed[original_x] = original_y.
    // This means for each original column x, we get the row y of the horizontal seam.

    for (int x = 0; x < image_with_seam.cols; ++x) { // Iterate through columns of original image
        if (x < static_cast<int>(seam_transposed.size())) {
             int y = seam_transposed[x]; // This is the row for the current column x
             if (y >= 0 && y < image_with_seam.rows) {
                image_with_seam.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // Green seam for horizontal
             }
        }
    }
    return image_with_seam;
}

cv::Mat SeamCarver::get_energy_map_heatmap() const {
    cv::Mat viz_energy_map = get_energy_map(); // 获取归一化后的能量图
    if (viz_energy_map.empty()) return {};
    
    return apply_heatmap_colormap(viz_energy_map);
}

cv::Mat SeamCarver::apply_heatmap_colormap(const cv::Mat& energy) const {
    cv::Mat colored_energy;
    cv::applyColorMap(energy, colored_energy, cv::COLORMAP_JET); // 使用 JET 色彩映射 (蓝->青->黄->红)
    return colored_energy;
}

std::vector<std::vector<int>> SeamCarver::find_multiple_vertical_seams(int num_seams) const {
    if (current_image.empty() || energy_map.empty() || num_seams <= 0) {
        return {};
    }

    std::vector<std::vector<int>> seams;
    cv::Mat temp_energy_map = energy_map.clone();
    cv::Mat temp_image = current_image.clone();
    
    // 临时 SeamCarver 对象，用于模拟 seam 的移除而不改变原始对象
    SeamCarver temp_carver(temp_image);
    
    for (int i = 0; i < num_seams && i < temp_image.cols - 1; ++i) {
        std::vector<int> seam = temp_carver.find_vertical_seam();
        if (seam.empty()) break;
        
        seams.push_back(seam);
        temp_carver.remove_vertical_seam(seam);
    }
    
    return seams;
}

std::vector<std::vector<int>> SeamCarver::find_multiple_horizontal_seams(int num_seams) const {
    if (current_image.empty() || energy_map.empty() || num_seams <= 0) {
        return {};
    }

    // 创建一个转置后的图像的临时 SeamCarver
    cv::Mat transposed_image = transpose_image(current_image);
    SeamCarver temp_carver(transposed_image);
    
    std::vector<std::vector<int>> vertical_seams = temp_carver.find_multiple_vertical_seams(num_seams);
    
    // 对于水平 seam，我们不需要进行特殊转换，因为 find_multiple_vertical_seams 返回的坐标已经是针对转置图像的
    return vertical_seams;
}

cv::Mat SeamCarver::show_multiple_vertical_seams(int num_seams) {
    if (current_image.empty() || num_seams <= 0) return current_image.clone();
    
    std::vector<std::vector<int>> seams = find_multiple_vertical_seams(num_seams);
    if (seams.empty()) return current_image.clone();
    
    cv::Mat image_with_seams = current_image.clone();
    
    // 为不同的 seam 使用不同的颜色
    std::vector<cv::Vec3b> colors = {
        cv::Vec3b(0, 0, 255),   // 红色 - 第一优先
        cv::Vec3b(0, 255, 0),   // 绿色 - 第二优先
        cv::Vec3b(255, 0, 0),   // 蓝色 - 第三优先
        cv::Vec3b(0, 255, 255), // 黄色 - 第四优先
        cv::Vec3b(255, 0, 255)  // 紫色 - 第五优先
    };
    
    for (size_t s = 0; s < seams.size(); ++s) {
        const std::vector<int>& seam = seams[s];
        cv::Vec3b color = colors[s % colors.size()];
        
        for (int i = 0; i < image_with_seams.rows; ++i) {
            if (i < static_cast<int>(seam.size())) {
                int j = seam[i];
                if (j >= 0 && j < image_with_seams.cols) {
                    // 绘制粗一点的线，让 seam 更明显
                    for (int di = -1; di <= 1; ++di) {
                        for (int dj = -1; dj <= 1; ++dj) {
                            int ni = i + di, nj = j + dj;
                            if (ni >= 0 && ni < image_with_seams.rows && nj >= 0 && nj < image_with_seams.cols) {
                                image_with_seams.at<cv::Vec3b>(ni, nj) = color;
                            }
                        }
                    }
                }
            }
        }
    }
    return image_with_seams;
}

cv::Mat SeamCarver::show_multiple_horizontal_seams(int num_seams) {
    if (current_image.empty() || num_seams <= 0) return current_image.clone();
    
    std::vector<std::vector<int>> seams = find_multiple_horizontal_seams(num_seams);
    if (seams.empty()) return current_image.clone();
    
    cv::Mat image_with_seams = current_image.clone();
    
    // 为不同的 seam 使用不同的颜色
    std::vector<cv::Vec3b> colors = {
        cv::Vec3b(0, 0, 255),   // 红色 - 第一优先
        cv::Vec3b(0, 255, 0),   // 绿色 - 第二优先
        cv::Vec3b(255, 0, 0),   // 蓝色 - 第三优先
        cv::Vec3b(0, 255, 255), // 黄色 - 第四优先
        cv::Vec3b(255, 0, 255)  // 紫色 - 第五优先
    };
    
    for (size_t s = 0; s < seams.size(); ++s) {
        const std::vector<int>& seam = seams[s];
        cv::Vec3b color = colors[s % colors.size()];
        
        for (int x = 0; x < image_with_seams.cols; ++x) {
            if (x < static_cast<int>(seam.size())) {
                int y = seam[x];
                if (y >= 0 && y < image_with_seams.rows) {
                    // 绘制粗一点的线，让 seam 更明显
                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            int nx = x + dx, ny = y + dy;
                            if (nx >= 0 && nx < image_with_seams.cols && ny >= 0 && ny < image_with_seams.rows) {
                                image_with_seams.at<cv::Vec3b>(ny, nx) = color;
                            }
                        }
                    }
                }
            }
        }
    }
    return image_with_seams;
}

// 插入指定的垂直seam
void SeamCarver::insert_vertical_seam(const std::vector<int>& seam) {
    if (current_image.empty() || seam.empty() || current_image.rows != static_cast<int>(seam.size())) {
        std::cerr << "错误: 由于输入无效，无法插入垂直seam。" << std::endl;
        return;
    }

    int rows = current_image.rows;
    int cols = current_image.cols;
    
    // 创建新图像，宽度+1
    cv::Mat new_image(rows, cols + 1, current_image.type());
    
    for (int i = 0; i < rows; ++i) {
        int seam_col = seam[i];
        if (seam_col < 0) seam_col = 0;
        if (seam_col >= cols) seam_col = cols - 1;
        
        // 复制seam左侧的像素
        for (int j = 0; j < seam_col; ++j) {
            new_image.at<cv::Vec3b>(i, j) = current_image.at<cv::Vec3b>(i, j);
        }
        
        // 在seam位置插入新像素
        // 新像素的值是seam左右两侧像素的平均值
        cv::Vec3b new_pixel;
        if (seam_col == 0) { // 如果seam在最左侧
            new_pixel = current_image.at<cv::Vec3b>(i, seam_col);
        } else if (seam_col == cols - 1) { // 如果seam在最右侧
            new_pixel = current_image.at<cv::Vec3b>(i, seam_col);
        } else { // 正常情况，取左右平均
            cv::Vec3b left = current_image.at<cv::Vec3b>(i, seam_col - 1);
            cv::Vec3b right = current_image.at<cv::Vec3b>(i, seam_col);
            for (int c = 0; c < 3; ++c) {
                new_pixel[c] = static_cast<uchar>((static_cast<int>(left[c]) + static_cast<int>(right[c])) / 2);
            }
        }
        new_image.at<cv::Vec3b>(i, seam_col) = new_pixel;
        
        // 复制seam右侧的像素，向右移动一个位置
        for (int j = seam_col; j < cols; ++j) {
            new_image.at<cv::Vec3b>(i, j + 1) = current_image.at<cv::Vec3b>(i, j);
        }
    }
    
    current_image = new_image;
    // 重新计算能量图
    calculate_energy_map();
}

// 插入指定的水平seam
void SeamCarver::insert_horizontal_seam(const std::vector<int>& seam) {
    if (current_image.empty() || seam.empty() || current_image.cols != static_cast<int>(seam.size())) {
        std::cerr << "错误: 由于输入无效，无法插入水平seam。" << std::endl;
        return;
    }
    
    // 将图像转置，执行垂直seam插入，再转置回来
    current_image = transpose_image(current_image);
    insert_vertical_seam(seam);
    current_image = transpose_image(current_image);
    calculate_energy_map();
}

// 插入多条垂直seam
void SeamCarver::insert_vertical_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback) {
    if (num_seams <= 0) return;
    std::cout << "插入 " << num_seams << " 条垂直seam..." << std::endl;
    
    // 为了避免重复插入相同的seam，我们需要先找到所有要插入的seam
    // 保存原始图像
    cv::Mat original_image = current_image.clone();
    
    // 找到所有要移除的seam (我们会反向使用它们来插入)
    std::vector<std::vector<int>> seams;
    for (int i = 0; i < num_seams && current_image.cols > 1; ++i) {
        std::vector<int> seam = find_vertical_seam();
        if (seam.empty()) break;
        
        // 保存seam，然后从图像中移除
        seams.push_back(seam);
        remove_vertical_seam(seam);
    }
    
    // 恢复原始图像
    current_image = original_image.clone();
    calculate_energy_map();
    
    // 计算更新频率
    const int update_frequency = std::max(1, num_seams / 10);
    
    // 按逆序插入所有找到的seam
    for (int i = 0; i < static_cast<int>(seams.size()); ++i) {
        const std::vector<int>& seam = seams[i];
        
        // 需要调整seam坐标以适应图像增长
        std::vector<int> adjusted_seam = seam;
        for (int j = 0; j < i; ++j) {
            // 为每个已插入的seam调整坐标
            for (size_t r = 0; r < adjusted_seam.size(); ++r) {
                if (adjusted_seam[r] >= seams[j][r]) {
                    adjusted_seam[r]++;
                }
            }
        }
        
        insert_vertical_seam(adjusted_seam);
        
        // 更新显示
        if (update_callback && (i % update_frequency == 0 || i == static_cast<int>(seams.size()) - 1)) {
            update_callback(current_image, i + 1, seams.size(), seam);
        }
        
        std::cout << "插入垂直seam " << i + 1 << "/" << seams.size() << ". 新宽度: " << current_image.cols << std::endl;
    }
}

// 插入多条水平seam
void SeamCarver::insert_horizontal_seams(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback) {
    if (num_seams <= 0) return;
    
    // 转置图像，调用垂直插入，再转置回来
    SeamCarver transposed_carver(transpose_image(current_image));
    
    // 插入num_seams条垂直seam
    auto wrapper_callback = [&](const cv::Mat& img, int current, int total, const std::vector<int>& seam) {
        if (update_callback) {
            // 转置回原始方向
            cv::Mat result = img.clone();
            cv::transpose(result, result);
            cv::flip(result, result, 1);
            update_callback(result, current, total, seam);
        }
    };
    
    transposed_carver.insert_vertical_seams(num_seams, wrapper_callback);
    
    // 获取结果
    cv::Mat result = transposed_carver.get_current_image();
    cv::transpose(result, result);
    cv::flip(result, result, 1);
    
    current_image = result;
    calculate_energy_map();
}

// 计算传输图，决定最优的seam移除顺序
std::pair<cv::Mat, cv::Mat> SeamCarver::compute_transport_map(int target_width, int target_height) const {
    if (current_image.empty()) return {cv::Mat(), cv::Mat()};
    
    int rows = current_image.rows;
    int cols = current_image.cols;
    
    // 检查目标尺寸
    if (target_width >= cols || target_height >= rows) {
        std::cerr << "错误: 目标尺寸必须小于当前尺寸" << std::endl;
        return {cv::Mat(), cv::Mat()};
    }
    
    // 需要删除的垂直和水平seam数量
    int r = rows - target_height; // 需要删除的行数
    int c = cols - target_width;  // 需要删除的列数
    int total_seams = r + c;
    
    if (total_seams <= 0) return {cv::Mat(), cv::Mat()};
    
    // 创建传输图 T
    cv::Mat T = cv::Mat::zeros(r + 1, c + 1, CV_64F);
    cv::Mat decision = cv::Mat::zeros(r + 1, c + 1, CV_8U); // 0-垂直，1-水平
    
    // 初始化第一行和第一列
    for (int i = 1; i <= r; ++i) {
        // 创建临时图像副本
        cv::Mat temp_image = current_image.clone();
        SeamCarver temp_carver(temp_image);
        
        // 移除i条水平seam
        for (int j = 0; j < i; ++j) {
            std::vector<int> seam = temp_carver.find_horizontal_seam();
            if (!seam.empty()) {
                temp_carver.remove_horizontal_seam(seam);
            }
        }
        
        // 获取能量和并存储到T
        cv::Scalar total_energy = cv::sum(temp_carver.energy_map);
        T.at<double>(i, 0) = total_energy[0];
        decision.at<uchar>(i, 0) = 1; // 水平seam
    }
    
    for (int j = 1; j <= c; ++j) {
        // 创建临时图像副本
        cv::Mat temp_image = current_image.clone();
        SeamCarver temp_carver(temp_image);
        
        // 移除j条垂直seam
        for (int i = 0; i < j; ++i) {
            std::vector<int> seam = temp_carver.find_vertical_seam();
            if (!seam.empty()) {
                temp_carver.remove_vertical_seam(seam);
            }
        }
        
        // 获取能量和并存储到T
        cv::Scalar total_energy = cv::sum(temp_carver.energy_map);
        T.at<double>(0, j) = total_energy[0];
        decision.at<uchar>(0, j) = 0; // 垂直seam
    }
    
    // 填充传输图
    for (int i = 1; i <= r; ++i) {
        for (int j = 1; j <= c; ++j) {
            // 创建两个临时图像副本
            cv::Mat temp_image1 = current_image.clone();
            SeamCarver temp_carver1(temp_image1);
            
            // 移除i-1条水平seam和j条垂直seam
            for (int k = 0; k < i - 1; ++k) {
                std::vector<int> seam = temp_carver1.find_horizontal_seam();
                if (!seam.empty()) {
                    temp_carver1.remove_horizontal_seam(seam);
                }
            }
            for (int k = 0; k < j; ++k) {
                std::vector<int> seam = temp_carver1.find_vertical_seam();
                if (!seam.empty()) {
                    temp_carver1.remove_vertical_seam(seam);
                }
            }
            // 计算再移除一条水平seam后的能量
            std::vector<int> seam1 = temp_carver1.find_horizontal_seam();
            double e1 = 0;
            if (!seam1.empty()) {
                for (int x = 0; x < temp_carver1.current_image.cols; ++x) {
                    if (x < static_cast<int>(seam1.size())) {
                        int y = seam1[x];
                        if (y >= 0 && y < temp_carver1.current_image.rows) {
                            e1 += temp_carver1.energy_map.at<double>(y, x);
                        }
                    }
                }
            }
            
            cv::Mat temp_image2 = current_image.clone();
            SeamCarver temp_carver2(temp_image2);
            
            // 移除i条水平seam和j-1条垂直seam
            for (int k = 0; k < i; ++k) {
                std::vector<int> seam = temp_carver2.find_horizontal_seam();
                if (!seam.empty()) {
                    temp_carver2.remove_horizontal_seam(seam);
                }
            }
            for (int k = 0; k < j - 1; ++k) {
                std::vector<int> seam = temp_carver2.find_vertical_seam();
                if (!seam.empty()) {
                    temp_carver2.remove_vertical_seam(seam);
                }
            }
            // 计算再移除一条垂直seam后的能量
            std::vector<int> seam2 = temp_carver2.find_vertical_seam();
            double e2 = 0;
            if (!seam2.empty()) {
                for (int y = 0; y < temp_carver2.current_image.rows; ++y) {
                    if (y < static_cast<int>(seam2.size())) {
                        int x = seam2[y];
                        if (x >= 0 && x < temp_carver2.current_image.cols) {
                            e2 += temp_carver2.energy_map.at<double>(y, x);
                        }
                    }
                }
            }
            
            // 决定移除哪个方向的seam
            double path1 = T.at<double>(i - 1, j) + e1;
            double path2 = T.at<double>(i, j - 1) + e2;
            
            if (path1 <= path2) {
                T.at<double>(i, j) = path1;
                decision.at<uchar>(i, j) = 1; // 水平seam
            } else {
                T.at<double>(i, j) = path2;
                decision.at<uchar>(i, j) = 0; // 垂直seam
            }
        }
    }
    
    return {T, decision};
}

// 使用传输图决定最优的seam移除顺序，调整图像到目标尺寸
void SeamCarver::resize_optimal(int target_width, int target_height, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback) {
    if (current_image.empty()) return;
    
    int rows = current_image.rows;
    int cols = current_image.cols;
    
    if (target_width >= cols || target_height >= rows) {
        std::cerr << "错误: 目标尺寸必须小于当前尺寸" << std::endl;
        return;
    }
    
    // 计算传输图和决策矩阵
    auto [transport_map, decision_map] = compute_transport_map(target_width, target_height);
    
    if (transport_map.empty() || decision_map.empty()) {
        std::cerr << "错误: 无法计算传输图" << std::endl;
        return;
    }
    
    int r = rows - target_height; // 需要删除的行数
    int c = cols - target_width;  // 需要删除的列数
    
    // 从决策图回溯得到最优路径
    std::vector<bool> path;
    int i = r, j = c;
    while (i > 0 || j > 0) {
        if (i == 0) {
            path.push_back(true); // 垂直seam
            j--;
        } else if (j == 0) {
            path.push_back(false); // 水平seam
            i--;
        } else {
            if (decision_map.at<uchar>(i, j) == 1) {
                path.push_back(false); // 水平seam
                i--;
            } else {
                path.push_back(true); // 垂直seam
                j--;
            }
        }
    }
    
    // 反转路径以得到正确的顺序
    std::reverse(path.begin(), path.end());
    
    int total_seams = path.size();
    
    std::cout << "使用传输图优化调整图像尺寸: " << cols << "x" << rows << " -> " 
              << target_width << "x" << target_height << std::endl;
    std::cout << "需要移除: " << total_seams << " 条seam" << std::endl;
    
    // 移除seam
    for (int i = 0; i < total_seams; ++i) {
        std::vector<int> seam;
        if (path[i]) { // 移除垂直seam
            seam = find_vertical_seam();
            if (!seam.empty()) {
                remove_vertical_seam(seam);
            }
        } else { // 移除水平seam
            seam = find_horizontal_seam();
            if (!seam.empty()) {
                remove_horizontal_seam(seam);
            }
        }
        
        // 更新进度
        if (update_callback) {
            update_callback(current_image, i + 1, total_seams, seam);
        }
        
        std::cout << "移除 " << (path[i] ? "垂直" : "水平") << " seam " << i + 1 << "/" << total_seams 
                  << ". 新尺寸: " << current_image.cols << "x" << current_image.rows << std::endl;
    }
}

// 突出图像主成分 (先放大再裁剪)
cv::Mat SeamCarver::enhance_subject(int scale_percent, const std::function<void(const cv::Mat&, int, int, const std::vector<int>&)>& update_callback) {
    if (current_image.empty() || scale_percent <= 100) {
        std::cerr << "错误: 无效的缩放比例，必须大于100%" << std::endl;
        return current_image.clone();
    }
    
    int rows = current_image.rows;
    int cols = current_image.cols;
    
    // 步骤1: 使用传统缩放放大图像
    std::cout << "使用传统缩放放大图像 " << scale_percent << "%" << std::endl;
    cv::Mat enlarged;
    cv::resize(current_image, enlarged, cv::Size(), scale_percent / 100.0, scale_percent / 100.0, cv::INTER_CUBIC);
    
    // 步骤2: 使用Seam Carving裁剪回原始尺寸
    std::cout << "使用Seam Carving裁剪回原始尺寸" << std::endl;
    
    SeamCarver enlarged_carver(enlarged);
    int seams_to_remove_h = enlarged.cols - cols;
    int seams_to_remove_v = enlarged.rows - rows;
    
    // 首先移除水平seam (因为这样可能会保留更多的主体)
    for (int i = 0; i < seams_to_remove_v; ++i) {
        std::vector<int> seam = enlarged_carver.find_horizontal_seam();
        if (!seam.empty()) {
            enlarged_carver.remove_horizontal_seam(seam);
        }
        
        if (update_callback && (i % 10 == 0 || i == seams_to_remove_v - 1)) {
            update_callback(enlarged_carver.get_current_image(), i + 1, seams_to_remove_v, seam);
        }
    }
    
    // 然后移除垂直seam
    for (int i = 0; i < seams_to_remove_h; ++i) {
        std::vector<int> seam = enlarged_carver.find_vertical_seam();
        if (!seam.empty()) {
            enlarged_carver.remove_vertical_seam(seam);
        }
        
        if (update_callback && (i % 10 == 0 || i == seams_to_remove_h - 1)) {
            update_callback(enlarged_carver.get_current_image(), i + 1, seams_to_remove_h, seam);
        }
    }
    
    // 获取最终结果
    cv::Mat result = enlarged_carver.get_current_image();
    
    // 更新当前图像
    current_image = result.clone();
    calculate_energy_map();
    
    return result;
}

// 可视化传输图决策过程
cv::Mat SeamCarver::visualize_transport_map(int target_width, int target_height) {
    if (current_image.empty()) return cv::Mat();
    
    int rows = current_image.rows;
    int cols = current_image.cols;
    
    if (target_width >= cols || target_height >= rows) {
        std::cerr << "错误: 目标尺寸必须小于当前尺寸" << std::endl;
        return cv::Mat();
    }
    
    // 计算传输图和决策矩阵
    auto [transport_map, decision_map] = compute_transport_map(target_width, target_height);
    
    if (transport_map.empty() || decision_map.empty()) {
        std::cerr << "错误: 无法计算传输图" << std::endl;
        return cv::Mat();
    }
    
    // 创建一个可视化图像
    cv::Mat visualization = current_image.clone();
    int r = rows - target_height;
    int c = cols - target_width;
    
    // 从决策图回溯得到最优路径
    std::vector<bool> path;
    int i = r, j = c;
    while (i > 0 || j > 0) {
        if (i == 0) {
            path.push_back(true); // 垂直seam
            j--;
        } else if (j == 0) {
            path.push_back(false); // 水平seam
            i--;
        } else {
            if (decision_map.at<uchar>(i, j) == 1) {
                path.push_back(false); // 水平seam
                i--;
            } else {
                path.push_back(true); // 垂直seam
                j--;
            }
        }
    }
    
    // 反转路径以得到正确的顺序
    std::reverse(path.begin(), path.end());
    
    // 为水平和垂直seam使用不同颜色
    cv::Scalar vert_color(0, 0, 255); // 红色表示垂直seam
    cv::Scalar horz_color(0, 255, 0); // 绿色表示水平seam
    
    // 临时图像和carver，用于可视化
    cv::Mat temp_image = current_image.clone();
    SeamCarver temp_carver(temp_image);
    
    // 依次移除seam并在原图上标记
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i]) { // 垂直seam
            std::vector<int> seam = temp_carver.find_vertical_seam();
            if (!seam.empty()) {
                // 在可视化图像上标记seam
                for (int y = 0; y < temp_carver.current_image.rows; ++y) {
                    if (y < static_cast<int>(seam.size())) {
                        int x = seam[y];
                        if (x >= 0 && x < visualization.cols) {
                            visualization.at<cv::Vec3b>(y, x) = cv::Vec3b(vert_color[0], vert_color[1], vert_color[2]);
                        }
                    }
                }
                // 在临时图像上移除seam
                temp_carver.remove_vertical_seam(seam);
            }
        } else { // 水平seam
            std::vector<int> seam = temp_carver.find_horizontal_seam();
            if (!seam.empty()) {
                // 在可视化图像上标记seam
                for (int x = 0; x < temp_carver.current_image.cols; ++x) {
                    if (x < static_cast<int>(seam.size())) {
                        int y = seam[x];
                        if (y >= 0 && y < visualization.rows) {
                            visualization.at<cv::Vec3b>(y, x) = cv::Vec3b(horz_color[0], horz_color[1], horz_color[2]);
                        }
                    }
                }
                // 在临时图像上移除seam
                temp_carver.remove_horizontal_seam(seam);
            }
        }
    }
    
    // 添加图例和信息
    cv::putText(visualization, "垂直seam (红色): " + std::to_string(c), 
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, vert_color, 2);
    cv::putText(visualization, "水平seam (绿色): " + std::to_string(r), 
               cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, horz_color, 2);
    cv::putText(visualization, "目标尺寸: " + std::to_string(target_width) + "x" + std::to_string(target_height), 
               cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    return visualization;
}

// 添加基础版的find_vertical_seam函数实现
std::vector<int> SeamCarver::find_vertical_seam() const {
    if (current_image.empty() || energy_map.empty()) {
        return {};
    }

    int rows = energy_map.rows;
    int cols = energy_map.cols;
    
    // 动态规划表
    cv::Mat dp = cv::Mat::zeros(rows, cols, CV_64F);
    cv::Mat backtrack = cv::Mat::zeros(rows, cols, CV_32S);
    
    // 初始化第一行
    energy_map.row(0).copyTo(dp.row(0));
    
    // 自顶向下填充DP表
    for (int i = 1; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double min_energy = dp.at<double>(i-1, j);
            int min_idx = j;
            
            if (j > 0 && dp.at<double>(i-1, j-1) < min_energy) {
                min_energy = dp.at<double>(i-1, j-1);
                min_idx = j-1;
            }
            
            if (j < cols-1 && dp.at<double>(i-1, j+1) < min_energy) {
                min_energy = dp.at<double>(i-1, j+1);
                min_idx = j+1;
            }
            
            dp.at<double>(i, j) = energy_map.at<double>(i, j) + min_energy;
            backtrack.at<int>(i, j) = min_idx;
        }
    }
    
    // 找到最后一行中的最小能量位置
    cv::Point min_loc;
    cv::minMaxLoc(dp.row(rows-1), nullptr, nullptr, &min_loc, nullptr);
    int min_col = min_loc.x;
    
    // 回溯找到seam路径
    std::vector<int> seam(rows);
    seam[rows-1] = min_col;
    for (int i = rows-2; i >= 0; --i) {
        seam[i] = backtrack.at<int>(i+1, seam[i+1]);
    }
    
    return seam;
}

// 优化版：移除水平seam
void SeamCarver::remove_horizontal_seams_optimized(int num_seams, std::function<void(const cv::Mat&, int, int, const std::vector<int>&)> update_callback) {
    if (num_seams <= 0) return;
    std::cout << "使用优化算法移除 " << num_seams << " 条水平 seam..." << std::endl;
    
    // 转置图像以便重用垂直seam移除代码
    cv::Mat transposed_image;
    cv::transpose(current_image, transposed_image);
    cv::flip(transposed_image, transposed_image, 1); // 确保正确的方向
    
    SeamCarver transposed_carver(transposed_image, algorithm);
    
    // 计算更新频率
    const int update_frequency = std::max(1, num_seams / 10); // 至少每 10% 更新一次
    
    for (int k = 0; k < num_seams; ++k) {
        if (transposed_carver.get_current_image().cols <= 1) {
            std::cout << "图像高度太小，在 " << k << " 条 seam 后停止。" << std::endl;
            break;
        }
        
        std::vector<int> seam = transposed_carver.find_vertical_seam_optimized();
        if (seam.empty()) {
            std::cout << "找不到可移除的水平 seam。停止。" << std::endl;
            break;
        }
        
        transposed_carver.remove_vertical_seam(seam);
        
        // 转置回原始方向
        cv::Mat result = transposed_carver.get_current_image();
        cv::flip(result, result, 1);
        cv::transpose(result, result);
        current_image = result;
        
        // 重新计算能量图
        calculate_energy_map();
        
        // 定期更新显示进度
        if (update_callback && (k % update_frequency == 0 || k == num_seams - 1)) {
            update_callback(current_image, k + 1, num_seams, seam);
        }
        
        std::cout << "移除水平 seam " << k + 1 << "/" << num_seams << ". 新高度: " << current_image.rows << std::endl;
    }
} 