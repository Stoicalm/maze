# Maze Path Planning System - 迷宫路径规划系统

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## 项目简介
本项目使用强化学习Q-learning算法和传统A*算法解决迷宫最短路径问题。系统能够从迷宫图片自动识别结构，训练智能体寻找最优路径，并提供直观的GUI界面进行交互操作。

## 📁 文件结构
maze-project/
├── maze_process.py # 迷宫图像处理模块
├── q-learning.py # Q-learning算法和GUI界面
├── requirements.txt # 项目依赖包
├── maze.jpg # 迷宫图片（示例）
├── README.md # 项目说明文档
└── maze_training_matrix.npy # 生成的迷宫矩阵（运行时创建）

## 🚀 快速开始

### 1. 环境准备
确保已安装Python 3.8或更高版本，然后安装依赖：
```bash
pip install -r requirements.txt

python maze_process.py

python q-learning.py

