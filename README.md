# 图像无缝裁剪系统 (Seam Carving)

## 项目简介

图像无缝裁剪（Seam Carving）是一种内容感知的图像大小调整技术，能够智能地移除图像中的低能量像素路径（即"缝线"），同时保留图像中的主要内容。与传统的裁剪或缩放方法不同，无缝裁剪可以更好地保持图像中的重要特征和结构。

本项目实现了多种无缝裁剪算法：
- **基本算法**：使用能量函数找出图像中能量最低的像素路径进行移除
- **优化算法**：采用优化的动态规划方法，提高了处理速度
- **前向能量算法**：考虑像素移除后产生的新边界能量，减少视觉伪影

## 功能特性

- 🖼️ 图像智能缩小：保留主要内容的同时减小图像尺寸
- 🔍 图像智能放大：通过插入新的缝线来放大图像
- 🎯 主体增强：突出图像中的重要内容
- 🔄 内容感知裁剪：比传统裁剪更智能地调整图像比例
- 📊 可视化：展示能量图、缝线位置和处理过程
- 🎬 过程动画：生成展示缝线移除过程的视频

## 算法原理

无缝裁剪算法基于以下步骤：

1. **能量计算**：计算图像中每个像素的能量值（通常基于像素梯度）
2. **缝线识别**：使用动态规划找到能量最低的连续像素路径（缝线）
3. **缝线移除/插入**：移除/插入缝线以调整图像大小

本项目实现了三种能量计算方法：
- 基本梯度能量
- 优化的累积能量计算
- 前向能量（考虑像素移除后产生的新边界）

## 安装指南

### 前提条件

- C++17兼容的编译器
- CMake (≥ 3.10)
- OpenCV库

### 编译步骤

```bash
# 克隆仓库
git clone https://github.com/your-username/SeamCarving.git
cd SeamCarving

# 创建构建目录
mkdir build && cd build

# 配置并编译
cmake ..
make

# 运行程序
./bin/SeamCarving
```

## 使用说明

### 基本用法

```bash
./bin/SeamCarving <输入图像> <目标宽度> <目标高度> [选项]
```

### 选项

- `--algorithm <basic|optimized|forward>`: 选择使用的算法（默认：optimized）
- `--method <standard|optimal|traditional>`: 选择调整大小的方法（默认：standard）
- `--enlarge`: 放大图像而非缩小
- `--enhance <percentage>`: 通过主体增强来突出重要内容
- `--visualize`: 生成可视化结果
- `--output <文件夹>`: 指定输出文件夹路径

### 示例

```bash
# 将图像缩小到400x300，使用前向能量算法
./bin/SeamCarving test_images/Mountain_Valley_Scenery.jpg 400 300 --algorithm forward --visualize

# 将图像水平扩展30%，保持高度不变
./bin/SeamCarving test_images/sunset.jpeg 650 0 --enlarge --visualize

# 主体增强，突出图像中的主要内容
./bin/SeamCarving test_images/HJoceanSmall.png 0 0 --enhance 30 --visualize
```

## 参考文献

- Avidan, S., & Shamir, A. (2007). *Seam carving for content-aware image resizing.* ACM Transactions on Graphics (TOG), 26(3), 10.
- Rubinstein, M., Shamir, A., & Avidan, S. (2008). *Improved seam carving for video retargeting.* ACM Transactions on Graphics (TOG), 27(3), 16.