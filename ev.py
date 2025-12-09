import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob

# sklearn 库
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 你的自定义模块
from src.preprocessor import LidarPreprocessor
from src.spectral_math import SpectralExtractor
from src.classifier import ActivityRecognizer
from src.loader import load_bin_sequence 

# ==========================================
# 1. 配置部分
# ==========================================
# 数据集根目录 (请根据实际情况确认路径)
DATA_DIR = r"filtered_mmwave"

# 参数设置 (根据论文和实验调整)
SPECTRAL_RADIUS = 0.15   # ε-邻域半径
N_EIGENVALUES = 60       # 选取前 k 个特征值/向量
WINDOW_SIZE = 40         # 滑动窗口帧数
WINDOW_STRIDE = 5        # 滑动窗口步长
GROUND_THRESHOLD = -999.0 # 地面过滤阈值

# 动作标签映射 (如果需要特定的 ID 到名称映射，可以在此填充)
# 例如: {1: "Walking", 2: "Sitting", ...}
ACTION_LABELS = {} 

def evaluate_system():
    # 检查路径是否存在
    if not os.path.exists(DATA_DIR):
        print(f"Error: 路径不存在: {DATA_DIR}")
        return

    print(f"根目录: {DATA_DIR}")
    print("正在初始化处理模块...")
    
    # 初始化各个处理模块
    preprocessor = LidarPreprocessor(ground_height_threshold=GROUND_THRESHOLD)
    extractor = SpectralExtractor(radius=SPECTRAL_RADIUS, n_eigenvalues=N_EIGENVALUES)
    recognizer = ActivityRecognizer(window_size=WINDOW_SIZE)

    X_all = []
    y_all = []
    total_sequences = 0
    start_time = time.time()

    print("开始遍历数据集结构 [E -> S -> A] ...")

    # 1. 遍历环境 (Environment)
    env_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "E*")))
    for env_path in env_dirs:
        env_name = os.path.basename(env_path)
        
        # 2. 遍历受试者 (Subject)
        sub_dirs = sorted(glob.glob(os.path.join(env_path, "S*")))
        for sub_path in sub_dirs:
            sub_name = os.path.basename(sub_path)
            
            # 3. 遍历动作 (Action)
            act_dirs = sorted(glob.glob(os.path.join(sub_path, "A*")))
            for act_path in act_dirs:
                act_name = os.path.basename(act_path)
                
                # 解析动作ID (例如 "A01" -> 1)
                try:
                    action_id = int(act_name[1:])
                except ValueError:
                    continue # 跳过无法解析的文件夹

                # --- 加载点云序列 ---
                frames = load_bin_sequence(act_path)
                if not frames:
                    continue
                
                # --- 逐帧特征提取 ---
                sequence_features = []
                
                for points in frames:
                    # A. 预处理 (去噪/去地)
                    clean_points = preprocessor.get_person_cloud(points)
                    
                    # B. 谱特征提取
                    # 注意：这里接收两个返回值 (特征值, 特征向量)
                    eigvals, eigvecs = extractor.compute_features(clean_points)
                    
                    # C. 计算特征向量统计量 (论文核心数学部分)
                    # 特征向量具有符号任意性 (v 和 -v 等价)，因此先取绝对值
                    # axis=0 表示对所有点进行聚合，得到长度为 k 的向量
                    eigvec_means = np.mean(np.abs(eigvecs), axis=0) 
                    eigvec_stds  = np.std(np.abs(eigvecs), axis=0)
                    
                    # D. 构建单帧描述符
                    # 组合: [特征值(k), 特征向量均值(k), 特征向量标准差(k)]
                    # 最终维度 = 3 * N_EIGENVALUES
                    frame_descriptor = np.concatenate([eigvals, eigvec_means, eigvec_stds])
                    
                    sequence_features.append(frame_descriptor)

                # --- 时域滑动窗口聚合 ---
                # 如果序列长度小于窗口大小，则跳过
                if len(sequence_features) < WINDOW_SIZE:
                    continue
                    
                folder_vectors = []
                folder_labels = []
                
                # 计算窗口数量
                num_windows = (len(sequence_features) - WINDOW_SIZE) // WINDOW_STRIDE + 1
                
                for i in range(num_windows):
                    start_idx = i * WINDOW_STRIDE
                    end_idx = start_idx + WINDOW_SIZE
                    
                    # 获取当前窗口的所有帧特征
                    # shape: (WINDOW_SIZE, feature_dim)
                    win_arr = np.array(sequence_features[start_idx:end_idx])
                    
                    # E. 计算窗口统计特征 (Temporal Statistics)
                    # 将时序数据坍缩成一个固定长度的向量供分类器使用
                    feature_vector = np.concatenate([
                        np.mean(win_arr, axis=0),
                        np.std(win_arr, axis=0),
                        np.max(win_arr, axis=0),
                        np.min(win_arr, axis=0)
                    ])
                    
                    folder_vectors.append(feature_vector)
                    folder_labels.append(action_id)

                # 将当前文件夹提取的数据加入总集
                if folder_vectors:
                    X_all.extend(folder_vectors)
                    y_all.extend(folder_labels)
                    total_sequences += 1
                    
                # 打印进度 (覆盖同一行以保持整洁)
                print(f"正在处理: {env_name} / {sub_name} / {act_name} (ID={action_id}) | 总样本数: {len(X_all)}", end='\r')

    print(f"\n\n处理完成！耗时: {time.time()-start_time:.2f}s")
    print(f"共成功处理序列文件夹: {total_sequences}")
    
    if len(X_all) == 0:
        print("错误：未提取到任何数据。请检查文件夹结构或 .bin 文件。")
        return

    # 转换为 numpy 数组
    X = np.array(X_all)
    y = np.array(y_all)

    print(f"最终数据集形状: {X.shape}")
    print(f"包含动作类别: {sorted(list(set(y)))}")
    
    # 更新识别器的标签字典 (可选)
    recognizer.labels = ACTION_LABELS

    # ==========================================
    # 2. 训练与评估部分
    # ==========================================
    print("\n开始训练分类器 (Random Forest)...")
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    # 注意：这里我们直接使用 sklearn 的 RandomForest，而不是 ActivityRecognizer 的封装，
    # 这样可以更直观地看到评估过程。你也可以调用 recognizer.model.fit()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    # 打印结果
    print("\n=== 评估报告 ===")
    print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred))

# 运行主函数
if __name__ == "__main__":
    evaluate_system()