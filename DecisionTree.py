from Node import Node
import numpy as np
from collections import Counter
from graphviz import Digraph
import random
import os
os.environ["PATH"] += os.pathsep + 'C://Program Files//Graphviz//bin'
class DecisionTree:
    """
    Khởi tạo cây quyết định
    min_samples_split : số mẫu tối thiểu để phân chia nút
    max_depth : độ sâu tối đa của cây
    """
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
    
    """
    Hàm tính giá trị gini của 1 nút
    1 - (pi)^2
    """    
    @staticmethod
    def _gini(labels):      
        counter = Counter(labels)
        total_samples = len(labels)
        impurity = 1.0

        for label in counter:
            probability = counter[label] / total_samples
            impurity -= probability ** 2

        return impurity    
    """
    Tính toán sự phân chia tốt nhất
    """
    def _best_split(self, X, y):
        if y.size <= 1:
            return None
        best_split = {}
        best_gini = float('inf')
        n_rows, n_cols = X.shape
        
        """
        Duyệt từng thuộc tính của bộ dữ liệu
        """
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            thresholds = list(np.unique(X_curr))
            """
            Duyệt từng giá trị của thuộc tính đó
            """
            for i in range(0, len(thresholds)):
                threshold = thresholds[i]
                threshold_before = thresholds[i-1]
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])
                if len(df_left) > 0 and len(df_right) > 0:
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]
                    left_gini = self._gini(y_left)
                    right_gini = self._gini(y_right)
                    gini = (len(y_left) / len(y)) * left_gini + (len(y_right) / len(y)) * right_gini
                    if threshold == threshold_before:
                        continue
                    if gini < best_gini:
                        best_split = {
                            'feature_index' : f_idx,
                            'threshold': (threshold + threshold_before) / 2,
                            'data_left' : df_left,
                            'data_right' : df_right,
                            'gini': gini
                        }
                        best_gini = gini
        return best_split
    
    def _build(self, X, y, depth=0):
        """
        xây dựng cây quyết định theo phương pháp đệ quy dựa vảo chỉ số độ sâu
        """
        n_rows, n_cols = X.shape
        
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            best = self._best_split(X, y)
            if best is not None and len(best) > 0:
                left = self._build(
                    X=best['data_left'][: , :-1], 
                    y=best['data_left'][: , -1], 
                    depth=depth + 1
                )
                right = self._build(
                    X=best['data_right'][: , :-1], 
                    y=best['data_right'][: , -1], 
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'], 
                    threshold=best['threshold'], 
                    data_left=left, 
                    data_right=right, 
                    gini=best['gini']
                ) 
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )
    
    def fit(self, X, y):
        """
        Training tập dữ liệu đầu vào
        """
        self.root = self._build(X, y)
        
    def _predict(self, x, tree):
        
        """
        Duyệt cây đi từ gốc đến lá
        """
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)

        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.data_right)
        
    def predict(self, X):
        """
        Cho ra kết quả của tập test
        """
        return [self._predict(x, self.root) for x in X]
    
    def visualize_tree(self, name_tree):
        dot = Digraph()

        def traverse(node, parent_id):
            if node.value is not None:
                dot.node(str(id(node)), label=str(node.value), shape='box')
            else:
                dot.node(str(id(node)), label=f"Gini: {node.gini};Feature {node.feature}\n<= {node.threshold}")
                traverse(node.data_left, id(node))
                dot.edge(str(id(node)), str(id(node.data_left)), label='True')

                traverse(node.data_right, id(node))
                dot.edge(str(id(node)), str(id(node.data_right)), label='False')
        traverse(self.root, None)
        dot.render('decision_tree' + name_tree, format='png', view=True)