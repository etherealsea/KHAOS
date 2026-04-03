import numpy as np

class WalkForwardValidator:
    def __init__(self, n_splits=5, train_ratio=0.6, test_ratio=0.2, gap_ratio=0.0):
        """
        滚动窗口交叉验证
        Train | Test | ... (Next Fold)
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        
    def split(self, data_length):
        """
        生成 (train_start, train_end, test_start, test_end) 索引元组
        """
        indices = []
        fold_size = data_length // (self.n_splits + 1)
        
        # 简单的滚动方案：
        # Fold 1: Train [0:k], Test [k:k+m]
        # Fold 2: Train [m:m+k], Test [m+k:m+k+m]
        # 这里我们使用 "Expanding Window" 或者 "Rolling Window"
        # 采用 Rolling Window 以适应非平稳性
        
        window_size = int(data_length * self.train_ratio / 2) # 缩小一点以适应多次切分
        test_size = int(data_length * self.test_ratio / 2)
        step = (data_length - window_size - test_size) // self.n_splits
        
        if step <= 0:
            # 数据太少，退化为单次切分
            return [(0, int(data_length*0.7), int(data_length*0.7), data_length)]
            
        current_start = 0
        
        for i in range(self.n_splits):
            train_end = current_start + window_size
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > data_length:
                break
                
            indices.append((current_start, train_end, test_start, test_end))
            current_start += step
            
        return indices
