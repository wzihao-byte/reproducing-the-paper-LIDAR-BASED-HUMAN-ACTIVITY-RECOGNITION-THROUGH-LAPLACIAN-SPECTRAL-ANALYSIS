import numpy as np
from sklearn.ensemble import RandomForestClassifier

class ActivityRecognizer:
    def __init__(self, window_size=40):
        self.model = RandomForestClassifier(n_estimators=50)
        self.window_buffer = [] 
        self.is_trained = False
        self.labels = {} 
        self.window_size = window_size # 动态设置窗口大小

    def aggregate_window(self, window_features):
        """
        计算时间窗口内的统计特征 (Mean, Std, Max, Min)
        """
        arr = np.array(window_features)
        return np.concatenate([
            np.mean(arr, axis=0),
            np.std(arr, axis=0),
            np.max(arr, axis=0),
            np.min(arr, axis=0)
        ])

    def predict(self, frame_feature):
        """
        缓存帧并预测活动
        """
        if not self.is_trained:
            return "Initializing..."

        # 添加当前帧的特征值到缓冲区
        self.window_buffer.append(frame_feature)
        
        # 保持缓冲区大小固定为 window_size (例如 40)
        if len(self.window_buffer) > self.window_size:
            self.window_buffer.pop(0)

        # 只有当缓冲区填满时才进行预测
        if len(self.window_buffer) == self.window_size:
            # 1. 聚合时间统计特征
            window_summary = self.aggregate_window(self.window_buffer)
            
            # 2. 预测
            pred_idx = self.model.predict([window_summary])[0]
            
            # 尝试将 ID 转换为文本标签
            if self.labels and pred_idx in self.labels:
                return self.labels[pred_idx]
            return pred_idx
        else:
            return "Buffering..."