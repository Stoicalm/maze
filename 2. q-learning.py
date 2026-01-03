import numpy as np
import heapq
import random
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import sys  # 添加sys模块以便退出程序

# -------------------------- 强化学习迷宫求解器 --------------------------
class MazeSolverRL:
    """基于强化学习的迷宫最短路径求解器"""
    
    def __init__(self, maze_matrix, start_pos=None):
        """
        初始化迷宫求解器
        :param maze_matrix: 迷宫矩阵 (49×65)，0=路，1=墙，2=起点
        :param start_pos: 起点位置 (row, col)，如果为None则从矩阵中找值为2的位置
        """
        self.maze = maze_matrix.copy()
        self.rows, self.cols = maze_matrix.shape
        
        # 找到起点位置（值为2的位置）
        if start_pos is None:
            start_positions = np.where(maze_matrix == 2)
            if len(start_positions[0]) > 0:
                self.start = (start_positions[0][0], start_positions[1][0])
            else:
                self.start = (0, 0)
                print("警告：未找到起点，使用(0,0)作为起点")
        else:
            self.start = start_pos
        
        # 确保起点是通路
        self.maze[self.start[0], self.start[1]] = 0
        
        # Q表：存储每个状态-动作对的值
        self.q_table = {}
        
        # 学习参数（根据您提供的参数）
        self.alpha = 0.25  # 学习率
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 0.5  # 探索率
        
        # 动作空间
        self.actions = [0, 1, 2, 3]  # 0:上, 1:下, 2:左, 3:右
        self.action_directions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        # 训练历史
        self.training_history = []
        
        # 奖励设置（按照您的要求）
        self.rewards = {
            'goal': 1000,        # 到达终点
            'wall': -10000,      # 碰到墙（负无穷用-10000近似）
            'boundary': -10000,  # 超过边界
            'step': -1           # 每走一步
        }
    
    def get_reward(self, state, next_state):
        """根据状态计算奖励"""
        row, col = state
        next_row, next_col = next_state
        
        # 检查是否超过边界
        if (next_row < 0 or next_row >= self.rows or 
            next_col < 0 or next_col >= self.cols):
            return self.rewards['boundary'], state
        
        # 检查是否撞墙
        if self.maze[next_row, next_col] == 1:
            return self.rewards['wall'], state
        
        # 检查是否到达终点
        if hasattr(self, 'end') and (next_row, next_col) == self.end:
            return self.rewards['goal'], next_state
        
        # 正常移动
        return self.rewards['step'], next_state
    
    def get_q_value(self, state, action):
        """获取Q值，如果不存在则初始化为0"""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        return self.q_table[state][action]
    
    def get_valid_actions(self, state):
        """获取当前位置的有效动作"""
        row, col = state
        valid_actions = []
        
        for action in self.actions:
            dr, dc = self.action_directions[action]
            new_row, new_col = row + dr, col + dc
            
            # 检查是否出界或撞墙
            if (0 <= new_row < self.rows and 
                0 <= new_col < self.cols and 
                self.maze[new_row, new_col] == 0):
                valid_actions.append(action)
        
        return valid_actions
    
    def choose_action(self, state):
        """选择动作（ε-greedy策略）"""
        valid_actions = self.get_valid_actions(state)
        
        if not valid_actions:
            return None
        
        # 探索：随机选择动作
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # 利用：选择Q值最大的动作
        q_values = [self.get_q_value(state, a) for a in valid_actions]
        max_q = max(q_values)
        
        # 如果有多个相同最大Q值的动作，随机选择一个
        best_actions = [valid_actions[i] for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        """更新Q值：Q(s,a) = Q(s,a) + α * [r + γ * maxQ(s',a') - Q(s,a)]"""
        current_q = self.get_q_value(state, action)
        
        # 计算下一个状态的最大Q值
        next_actions = self.get_valid_actions(next_state)
        if next_actions:
            max_next_q = max([self.get_q_value(next_state, a) for a in next_actions])
        else:
            max_next_q = 0
        
        # Q-learning更新公式
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def train_one_episode(self):
        """训练一个episode"""
        state = self.start
        total_reward = 0
        steps = 0
        visited = set([state])
        
        while steps < self.max_steps:
            action = self.choose_action(state)
            
            if action is None:
                break
            
            # 执行动作
            dr, dc = self.action_directions[action]
            next_state = (state[0] + dr, state[1] + dc)
            
            # 获取奖励
            reward, actual_next_state = self.get_reward(state, next_state)
            
            # 更新Q值
            self.update_q_value(state, action, reward, actual_next_state)
            
            # 如果撞墙或出界，状态不变
            if reward == self.rewards['wall'] or reward == self.rewards['boundary']:
                next_state = state
            else:
                state = actual_next_state
            
            total_reward += reward
            steps += 1
            visited.add(state)
            
            # 检查是否到达终点
            if hasattr(self, 'end') and state == self.end:
                break
        
        return total_reward, steps, state == self.end
    
    def train(self, end_pos, episodes=2000, max_steps=1000):
        """训练智能体到达指定终点"""
        print(f"\n开始训练，目标终点: {end_pos}")
        print(f"参数: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
        
        self.end = end_pos
        self.max_steps = max_steps
        
        # 确保终点可达
        if not self.is_reachable(self.start, end_pos):
            print("警告：终点不可达！")
            return False
        
        success_count = 0
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            reward, steps, success = self.train_one_episode()
            episode_rewards.append(reward)
            episode_lengths.append(steps)
            
            if success:
                success_count += 1
            
            # 每200轮打印进度
            if (episode + 1) % 200 == 0:
                avg_reward = np.mean(episode_rewards[-200:])
                avg_steps = np.mean(episode_lengths[-200:])
                success_rate = success_count / min(200, episode+1)
                print(f"Episode {episode+1}/{episodes}, "
                      f"平均奖励: {avg_reward:.1f}, "
                      f"平均步数: {avg_steps:.1f}, "
                      f"成功率: {success_rate:.2%}")
                success_count = 0  # 重置计数
        
        print("训练完成！")
        
        # 保存训练历史
        self.training_history.append({
            'end_pos': end_pos,
            'episodes': episodes,
            'rewards': episode_rewards,
            'lengths': episode_lengths
        })
        
        return True
    
    def find_path(self):
        """使用训练好的Q表寻找路径"""
        if not hasattr(self, 'end'):
            print("错误：未设置终点！")
            return None
        
        state = self.start
        path = [state]
        total_reward = 0
        
        max_steps = self.rows * self.cols * 2
        
        for step in range(max_steps):
            if state == self.end:
                print(f"成功到达终点！总奖励: {total_reward}, 步数: {len(path)-1}")
                return path
            
            valid_actions = self.get_valid_actions(state)
            
            if not valid_actions:
                print("无路可走！")
                return None
            
            # 选择Q值最大的动作
            q_values = [self.get_q_value(state, a) for a in valid_actions]
            best_action = valid_actions[np.argmax(q_values)]
            
            # 执行动作
            dr, dc = self.action_directions[best_action]
            next_state = (state[0] + dr, state[1] + dc)
            
            # 获取奖励
            reward, actual_next_state = self.get_reward(state, next_state)
            
            # 如果撞墙或出界，状态不变
            if reward == self.rewards['wall'] or reward == self.rewards['boundary']:
                next_state = state
            
            # 检查是否进入死循环
            if len(path) > 10 and state == path[-5]:
                print("检测到循环，停止搜索")
                return None
            
            path.append(next_state)
            total_reward += reward
            state = next_state
        
        print(f"超过最大步数 {max_steps}，未找到路径")
        return None
    
    def find_shortest_path_astar(self, end_pos):
        """使用A*算法寻找最短路径（作为对比）"""
        print("使用A*算法寻找最短路径...")
        
        # 启发函数：曼哈顿距离
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # 初始化
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: heuristic(self.start, end_pos)}
        
        visited_count = 0
        
        while open_set:
            _, current = heapq.heappop(open_set)
            visited_count += 1
            
            if current == end_pos:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                path.reverse()
                
                print(f"A*算法找到路径！访问节点数: {visited_count}, 路径长度: {len(path)-1}")
                return path
            
            # 探索邻居
            for action in self.actions:
                dr, dc = self.action_directions[action]
                neighbor = (current[0] + dr, current[1] + dc)
                
                # 检查是否有效
                if (0 <= neighbor[0] < self.rows and 
                    0 <= neighbor[1] < self.cols and 
                    self.maze[neighbor[0], neighbor[1]] == 0):
                    
                    tentative_g_score = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, end_pos)
                        
                        # 检查是否已在open_set中
                        in_open_set = any(neighbor == item[1] for item in open_set)
                        if not in_open_set:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print("A*算法未找到路径")
        return None
    
    def is_reachable(self, start_pos, end_pos):
        """检查终点是否可达（使用BFS）"""
        if self.maze[end_pos[0], end_pos[1]] == 1:
            return False
        
        visited = np.zeros((self.rows, self.cols), dtype=bool)
        queue = [start_pos]
        visited[start_pos[0], start_pos[1]] = True
        
        while queue:
            row, col = queue.pop(0)
            
            if (row, col) == end_pos:
                return True
            
            for action in self.actions:
                dr, dc = self.action_directions[action]
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < self.rows and 
                    0 <= new_col < self.cols and 
                    not visited[new_row, new_col] and 
                    self.maze[new_row, new_col] == 0):
                    visited[new_row, new_col] = True
                    queue.append((new_row, new_col))
        
        return False
    
    def print_q_table_stats(self):
        """打印Q表统计信息"""
        total_states = len(self.q_table)
        total_actions = sum(len(actions) for actions in self.q_table.values())
        
        print(f"\nQ表统计:")
        print(f"  状态数: {total_states}")
        print(f"  状态-动作对: {total_actions}")
        
        # 计算Q值的统计信息
        all_q_values = []
        for state_actions in self.q_table.values():
            all_q_values.extend(state_actions.values())
        
        if all_q_values:
            print(f"  Q值范围: [{min(all_q_values):.2f}, {max(all_q_values):.2f}]")
            print(f"  Q值平均值: {np.mean(all_q_values):.2f}")

# -------------------------- 交互式GUI界面 --------------------------
class MazeSolverGUI:
    """交互式迷宫求解器GUI"""
    
    def __init__(self, maze_matrix):
        self.maze = maze_matrix
        self.rows, self.cols = maze_matrix.shape
        
        # 找到起点
        start_positions = np.where(maze_matrix == 2)
        if len(start_positions[0]) > 0:
            self.start_pos = (start_positions[0][0], start_positions[1][0])
        else:
            self.start_pos = (0, 0)
        
        # 初始化强化学习求解器
        self.solver = MazeSolverRL(maze_matrix, self.start_pos)
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("迷宫路径规划系统 - 强化学习")
        self.root.geometry("1200x800")
        
        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 创建左侧面板
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 迷宫信息
        info_frame = ttk.LabelFrame(left_frame, text="迷宫信息", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        info_text = f"迷宫尺寸: {self.rows}行 × {self.cols}列\n"
        info_text += f"起点位置: ({self.start_pos[0]}, {self.start_pos[1]})\n"
        info_text += f"墙壁数量: {np.sum(maze_matrix == 1)}\n"
        info_text += f"通道数量: {np.sum(maze_matrix == 0)}"
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack()
        
        # 参数设置
        param_frame = ttk.LabelFrame(left_frame, text="强化学习参数", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 学习率
        ttk.Label(param_frame, text="学习率 (α):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.alpha_var = tk.DoubleVar(value=0.25)
        self.alpha_entry = ttk.Entry(param_frame, textvariable=self.alpha_var, width=10)
        self.alpha_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 折扣因子
        ttk.Label(param_frame, text="折扣因子 (γ):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.gamma_var = tk.DoubleVar(value=0.95)
        self.gamma_entry = ttk.Entry(param_frame, textvariable=self.gamma_var, width=10)
        self.gamma_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 探索率
        ttk.Label(param_frame, text="探索率 (ε):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.epsilon_var = tk.DoubleVar(value=0.5)
        self.epsilon_entry = ttk.Entry(param_frame, textvariable=self.epsilon_var, width=10)
        self.epsilon_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 训练轮数
        ttk.Label(param_frame, text="训练轮数:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.episodes_var = tk.IntVar(value=2000)
        self.episodes_entry = ttk.Entry(param_frame, textvariable=self.episodes_var, width=10)
        self.episodes_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # 终点设置
        end_frame = ttk.LabelFrame(left_frame, text="终点设置", padding="10")
        end_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(end_frame, text="列 (0-{}):".format(self.cols-1)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.end_col_var = tk.IntVar(value=self.cols//2)
        self.end_col_entry = ttk.Entry(end_frame, textvariable=self.end_col_var, width=10)
        self.end_col_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(end_frame, text="行 (0-{}):".format(self.rows-1)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.end_row_var = tk.IntVar(value=self.rows//2)
        self.end_row_entry = ttk.Entry(end_frame, textvariable=self.end_row_var, width=10)
        self.end_row_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 按钮面板
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="检查终点可达性", 
                  command=self.check_reachable).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="训练智能体", 
                  command=self.train_agent).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Q-learning路径规划", 
                  command=self.find_path_qlearning).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="A*算法路径规划", 
                  command=self.find_path_astar).pack(side=tk.LEFT, padx=5)
        
        # 结果信息
        result_frame = ttk.LabelFrame(left_frame, text="结果信息", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(result_frame, height=15, width=50)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 初始化结果显示
        self.update_result("欢迎使用迷宫路径规划系统！\n")
        self.update_result(f"迷宫尺寸: {self.rows}×{self.cols}\n")
        self.update_result(f"起点位置: ({self.start_pos[0]}, {self.start_pos[1]})\n")
        self.update_result("\n请先设置终点位置和参数，然后训练智能体。\n")
        
        # 创建右侧面板（迷宫可视化）
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始绘制迷宫
        self.draw_maze()
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 在__init__方法中添加窗口关闭事件绑定
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def update_result(self, text):
        """更新结果文本框"""
        self.result_text.insert(tk.END, text)
        self.result_text.see(tk.END)
        self.result_text.update()
    
    def clear_result(self):
        """清空结果文本框"""
        self.result_text.delete(1.0, tk.END)
    
    def update_status(self, text):
        """更新状态栏"""
        self.status_var.set(text)
        self.root.update()
    
    def draw_maze(self, path=None):
        """绘制迷宫"""
        self.ax.clear()
        
        # 绘制墙壁（黑色）和通道（白色）
        maze_display = np.zeros((self.rows, self.cols, 3))
        maze_display[self.maze == 1] = [0, 0, 0]  # 黑色墙壁
        maze_display[self.maze == 0] = [1, 1, 1]  # 白色通道
        
        # 绘制起点（绿色）
        start_r, start_c = self.start_pos
        maze_display[start_r, start_c] = [0, 1, 0]  # 绿色起点
        
        # 绘制终点（蓝色）
        if hasattr(self, 'end_pos'):
            end_r, end_c = self.end_pos
            if 0 <= end_r < self.rows and 0 <= end_c < self.cols:
                maze_display[end_r, end_c] = [0, 0, 1]  # 蓝色终点
        
        # 绘制路径（红色）
        if path:
            for (r, c) in path:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    maze_display[r, c] = [1, 0, 0]  # 红色路径
        
        self.ax.imshow(maze_display, interpolation='nearest')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
    
    def check_reachable(self):
        """检查终点是否可达"""
        try:
            end_r = self.end_row_var.get()
            end_c = self.end_col_var.get()
            
            if not (0 <= end_r < self.rows and 0 <= end_c < self.cols):
                messagebox.showerror("错误", f"终点坐标超出范围！\n行: 0-{self.rows-1}\n列: 0-{self.cols-1}")
                return
            
            self.end_pos = (end_r, end_c)
            
            # 检查是否在墙上
            if self.maze[end_r, end_c] == 1:
                messagebox.showwarning("警告", "终点在墙上！请选择其他位置。")
                return
            
            # 检查可达性
            self.update_status("检查可达性...")
            reachable = self.solver.is_reachable(self.start_pos, self.end_pos)
            
            if reachable:
                messagebox.showinfo("可达性检查", f"终点 ({end_r}, {end_c}) 可达！")
                self.update_result(f"终点 ({end_r}, {end_c}) 可达\n")
            else:
                messagebox.showwarning("可达性检查", f"终点 ({end_r}, {end_c}) 不可达！")
                self.update_result(f"终点 ({end_r}, {end_c}) 不可达\n")
            
            # 更新迷宫显示
            self.draw_maze()
            self.update_status("就绪")
            
        except Exception as e:
            messagebox.showerror("错误", f"参数错误: {str(e)}")
            self.update_status("就绪")
    
    def train_agent(self):
        """训练智能体"""
        try:
            # 获取参数
            self.solver.alpha = self.alpha_var.get()
            self.solver.gamma = self.gamma_var.get()
            self.solver.epsilon = self.epsilon_var.get()
            episodes = self.episodes_var.get()
            
            end_r = self.end_row_var.get()
            end_c = self.end_col_var.get()
            
            if not (0 <= end_r < self.rows and 0 <= end_c < self.cols):
                messagebox.showerror("错误", f"终点坐标超出范围！")
                return
            
            self.end_pos = (end_r, end_c)
            
            # 检查是否在墙上
            if self.maze[end_r, end_c] == 1:
                messagebox.showerror("错误", "终点在墙上！")
                return
            
            # 检查可达性
            if not self.solver.is_reachable(self.start_pos, self.end_pos):
                messagebox.showerror("错误", "终点不可达！")
                return
            
            # 开始训练
            self.update_status("训练中...")
            self.clear_result()
            self.update_result(f"开始训练...\n")
            self.update_result(f"参数: α={self.solver.alpha}, γ={self.solver.gamma}, ε={self.solver.epsilon}\n")
            self.update_result(f"终点: ({end_r}, {end_c})\n")
            self.update_result(f"训练轮数: {episodes}\n")
            self.update_result("="*40 + "\n")
            
            start_time = time.time()
            success = self.solver.train(self.end_pos, episodes=episodes)
            end_time = time.time()
            
            if success:
                self.update_result(f"\n训练完成！耗时: {end_time-start_time:.2f}秒\n")
                self.update_result(f"最终参数: α={self.solver.alpha}, γ={self.solver.gamma}, ε={self.solver.epsilon}\n")
                
                # 显示Q表统计
                self.update_result("\nQ表统计:\n")
                total_states = len(self.solver.q_table)
                self.update_result(f"  学习到的状态数: {total_states}\n")
                
                # 显示训练历史
                if self.solver.training_history:
                    last_history = self.solver.training_history[-1]
                    rewards = last_history['rewards']
                    if len(rewards) > 0:
                        avg_reward = np.mean(rewards[-100:])
                        self.update_result(f"  最近100轮平均奖励: {avg_reward:.2f}\n")
            else:
                self.update_result("训练失败！终点不可达。\n")
            
            self.update_status("就绪")
            
        except Exception as e:
            messagebox.showerror("错误", f"训练失败: {str(e)}")
            self.update_status("就绪")
    
    def find_path_qlearning(self):
        """使用Q-learning寻找路径"""
        try:
            end_r = self.end_row_var.get()
            end_c = self.end_col_var.get()
            
            if not (0 <= end_r < self.rows and 0 <= end_c < self.cols):
                messagebox.showerror("错误", f"终点坐标超出范围！")
                return
            
            self.end_pos = (end_r, end_c)
            
            # 检查是否训练过
            if not hasattr(self.solver, 'q_table') or len(self.solver.q_table) == 0:
                messagebox.showwarning("警告", "请先训练智能体！")
                return
            
            # 寻找路径
            self.update_status("使用Q-learning规划路径...")
            self.update_result("\n使用Q-learning规划路径...\n")
            
            start_time = time.time()
            path = self.solver.find_path()
            end_time = time.time()
            
            if path:
                path_length = len(path) - 1
                self.update_result(f"找到路径！\n")
                self.update_result(f"路径长度: {path_length}步\n")
                self.update_result(f"计算时间: {end_time-start_time:.2f}秒\n")
                self.update_result(f"路径坐标:\n")
                
                # 显示部分路径坐标
                for i, (r, c) in enumerate(path[:10]):
                    self.update_result(f"  第{i}步: ({r}, {c})\n")
                
                if len(path) > 10:
                    self.update_result(f"  ... 还有{len(path)-10}步\n")
                
                # 绘制路径
                self.draw_maze(path)
            else:
                self.update_result("未找到路径！\n")
            
            self.update_status("就绪")
            
        except Exception as e:
            messagebox.showerror("错误", f"路径规划失败: {str(e)}")
            self.update_status("就绪")
    
    def find_path_astar(self):
        """使用A*算法寻找路径"""
        try:
            end_r = self.end_row_var.get()
            end_c = self.end_col_var.get()
            
            if not (0 <= end_r < self.rows and 0 <= end_c < self.cols):
                messagebox.showerror("错误", f"终点坐标超出范围！")
                return
            
            self.end_pos = (end_r, end_c)
            
            # 使用A*算法
            self.update_status("使用A*算法规划路径...")
            self.update_result("\n使用A*算法规划路径...\n")
            
            start_time = time.time()
            path = self.solver.find_shortest_path_astar(self.end_pos)
            end_time = time.time()
            
            if path:
                path_length = len(path) - 1
                self.update_result(f"找到最短路径！\n")
                self.update_result(f"路径长度: {path_length}步\n")
                self.update_result(f"计算时间: {end_time-start_time:.2f}秒\n")
                self.update_result(f"路径坐标:\n")
                
                # 显示部分路径坐标
                for i, (r, c) in enumerate(path[:10]):
                    self.update_result(f"  第{i}步: ({r}, {c})\n")
                
                if len(path) > 10:
                    self.update_result(f"  ... 还有{len(path)-10}步\n")
                
                # 绘制路径
                self.draw_maze(path)
            else:
                self.update_result("未找到路径！\n")
            
            self.update_status("就绪")
            
        except Exception as e:
            messagebox.showerror("错误", f"路径规划失败: {str(e)}")
            self.update_status("就绪")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()
    
    # 添加窗口关闭事件的回调函数
    def on_close(self):
        """窗口关闭事件回调"""
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            self.root.destroy()
            sys.exit()

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    # 加载您处理好的迷宫矩阵（假设已经保存为npy文件）
    try:
        maze_matrix = np.load("maze_training_matrix.npy")
        print(f"成功加载迷宫矩阵，尺寸: {maze_matrix.shape}")
        
        # 找到起点位置
        start_positions = np.where(maze_matrix == 2)
        if len(start_positions[0]) > 0:
            start_row, start_col = start_positions[0][0], start_positions[1][0]
            print(f"起点位置: 行={start_row}, 列={start_col}")
        
        # 运行交互式GUI
        gui = MazeSolverGUI(maze_matrix)
        gui.run()
        
    except FileNotFoundError:
        print("错误: 找不到迷宫矩阵文件 'maze_training_matrix.npy'")
        print("请先运行迷宫处理代码生成迷宫矩阵")
        
        # 如果找不到文件，创建一个示例迷宫（用于测试）
        print("\n创建示例迷宫用于测试...")
        rows, cols = 49, 65
        
        # 生成一个随机迷宫（60%是路，40%是墙）
        maze_matrix = np.random.choice([0, 1], size=(rows, cols), p=[0.6, 0.4])
        
        # 设置起点位置（中间位置）
        start_row, start_col = rows // 2, cols // 2
        maze_matrix[start_row, start_col] = 2
        
        # 确保边界是墙
        maze_matrix[0, :] = 1
        maze_matrix[-1, :] = 1
        maze_matrix[:, 0] = 1
        maze_matrix[:, -1] = 1
        
        print(f"创建示例迷宫，尺寸: {maze_matrix.shape}")
        print(f"起点位置: ({start_row}, {start_col})")
        
        # 运行交互式GUI
        gui = MazeSolverGUI(maze_matrix)
        gui.run()