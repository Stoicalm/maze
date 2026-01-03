import cv2
import numpy as np

def preprocess_maze_image(img_path):
    """
    读取迷宫图片，预处理（去噪、二值化），区分墙（黑色）和背景（白色）
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("图片路径错误，无法读取！")
    
    img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    return img, gray, binary

def remove_black_borders(binary, original_img):
    """
    删除图片四周的黑色部分。
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return original_img
    
    x, y, w, h = cv2.boundingRect(coords)
    cropped_img = original_img[y:y+h, x:x+w]

    return cropped_img

def find_yellow_start(img):
    """
    定位图片中的黄色区域（起点），返回像素坐标数组
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 调整HSV范围，增强对黄色的识别
    lower_yellow = np.array([20, 40, 30])
    upper_yellow = np.array([40, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 形态学操作，增强黄色区域
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    yellow_coords = np.where(mask > 0)
    y_coords = yellow_coords[0]
    x_coords = yellow_coords[1]
    
    return x_coords, y_coords

def get_base_unit():
    """
    返回基本单位（固定为8像素）
    """
    base_unit = 8
    return base_unit

def create_maze_matrix(binary, base_unit):
    """
    按8像素分块，将图片转化为迷宫矩阵
    """
    img_h, img_w = binary.shape
    maze_rows = img_h // base_unit
    maze_cols = img_w // base_unit
    
    maze_matrix = np.zeros((maze_rows, maze_cols), dtype=np.int32)
    
    for i in range(maze_rows):
        for j in range(maze_cols):
            # 提取当前8×8块的像素
            block = binary[i*base_unit : (i+1)*base_unit,
                           j*base_unit : (j+1)*base_unit]
            
            # 统计黑色像素（墙）的数量
            black_pixels = np.sum(block == 0)
            total_pixels = block.size
            
            # 若黑色像素占比>90%，判定为可通行（0），否则为墙（1）
            if black_pixels > total_pixels * 0.9:
                maze_matrix[i, j] = 0
            else:
                maze_matrix[i, j] = 1

    print(f"生成的迷宫矩阵尺寸：{maze_rows}行 × {maze_cols}列")
    return maze_matrix

def print_maze_info(maze_matrix, start_row, start_col):
    """
    打印迷宫矩阵信息
    """
    print(f"迷宫矩阵尺寸: {maze_matrix.shape}")
    print(f"起点位置: 行={start_row}, 列={start_col}")
    print(f"起点值: {maze_matrix[start_row, start_col]}")
    
    # 统计矩阵中的不同值
    unique, counts = np.unique(maze_matrix, return_counts=True)
    value_counts = dict(zip(unique, counts))
    print(f"矩阵值统计: {value_counts}")
    
    print("\n迷宫矩阵前15行×15列:")
    for i in range(min(15, maze_matrix.shape[0])):
        row_str = ""
        for j in range(min(15, maze_matrix.shape[1])):
            if i == start_row and j == start_col:
                row_str += "2 "  # 起点
            elif maze_matrix[i, j] == 1:
                row_str += "1 "  # 墙
            else:
                row_str += "0 "  # 路
        print(row_str)

if __name__ == "__main__":
    img_path = "maze.jpg"
    
    try:
        # 步骤1：预处理图片
        print("1. 预处理图片...")
        img, gray, binary = preprocess_maze_image(img_path)
        print(f"   原始图片尺寸: {img.shape}")
        
        # 步骤2：删除黑色边框
        print("2. 删除黑色边框...")
        img = remove_black_borders(binary, img)
        print(f"   裁剪后图片尺寸: {img.shape}")
        
        # 重新生成灰度图和二值化图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 步骤3：定位黄色起点
        print("3. 定位黄色起点...")
        x_coords, y_coords = find_yellow_start(img)
        
        if len(x_coords) == 0:
            print("警告：未检测到黄色起点！")
            # 使用图片中心作为起点
            start_point_y = img.shape[0] // 2
            start_point_x = img.shape[1] // 2
            print(f"   使用默认起点: X={start_point_x}, Y={start_point_y}")
        else:
            start_point_x = int(np.mean(x_coords))
            start_point_y = int(np.mean(y_coords))
            print(f"   检测到黄色起点: X={start_point_x}, Y={start_point_y}")
        
        # 步骤4：确定基本单位
        print("4. 确定基本单位...")
        base_unit = get_base_unit()
        print(f"   基本单位: {base_unit}像素")
        
        # 步骤5：分块生成训练矩阵
        print("5. 生成迷宫矩阵...")
        maze_matrix = create_maze_matrix(binary, base_unit)
        
        # 步骤6：标注起点
        print("6. 标注起点位置...")
        start_row = start_point_y // base_unit
        start_col = start_point_x // base_unit
        
        # 确保起点在矩阵范围内
        start_row = max(0, min(start_row, maze_matrix.shape[0] - 1))
        start_col = max(0, min(start_col, maze_matrix.shape[1] - 1))
        
        # 将起点位置设为2
        maze_matrix[start_row, start_col] = 2
        
        # 打印迷宫信息
        print("\n" + "="*50)
        print_maze_info(maze_matrix, start_row, start_col)
        
        # 保存矩阵到文件
        np.save("maze_training_matrix.npy", maze_matrix)
        print(f"\n迷宫矩阵已保存为: maze_training_matrix.npy")
        
        # 找到并打印所有值为2的位置
        start_positions = np.where(maze_matrix == 2)
        print(f"\n起点位置(行,列): {list(zip(start_positions[0], start_positions[1]))}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()