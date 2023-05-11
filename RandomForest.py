from DecisionTree import DecisionTree
from collections import Counter
import numpy as np
class RandomForest:
    """
    Khởi tạo một rừng ngẫu nhiên cây quyết định
    num_trees : số cây quyết định
    min_samples_split : số mẫu nhỏ nhất để chia nhánh
    max_depth : độ sâu lớn nhất cho cây quyết định    
    """
    def __init__(self, num_trees=25, min_samples_split=2, max_depth=5):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.decision_trees = []
        
    @staticmethod
    def _sample(X, y):
        """
        Lấy mẫu ngẫu nhiên 1 tập dataset cho cây quyết định
        """
        samples = np.random.choice(len(X), size = len(X), replace = True)
        return X[samples], y[samples]
        
    def fit(self, X, y):
        """
        Huấn luyện để tạo ra các cây quyết định từ các tập dữ liệu ngẫu nhiên
        """
        if len(self.decision_trees) > 0:
            self.decision_trees = []
            
        num_built = 0
        while num_built < self.num_trees:
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            _X, _y = self._sample(X, y)
            tree.fit(_X, _y)
            tree.visualize_tree(str(num_built))
            self.decision_trees.append(tree)
            num_built += 1
    
    def predict(self, X):
        """
        Lấy kết quả của từng cây quyết định rồi lấy kết quả đa số mà các cây quyết định trả về
        """
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
        y = np.swapaxes(a=y, axis1=0, axis2=1)
        predictions = []
        for preds in y:
            counter = Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        return predictions