"""
Java-style AI Script in Python
Demonstrates Object-Oriented AI concepts: classification, decision trees, and algorithms
"""

from typing import List, Dict, Any
import random

class DecisionNode:
    """Node in a decision tree"""
    
    def __init__(self, feature: str = None, threshold: float = None, 
                 left = None, right = None, value: str = None):
        self.feature = feature        # Feature to split on
        self.threshold = threshold    # Threshold value for split
        self.left = left              # Left subtree
        self.right = right            # Right subtree
        self.value = value            # Class value if leaf node


class DecisionTreeClassifier:
    """Simple Decision Tree Classifier for AI"""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X: List[List[float]], y: List[str]):
        """Build decision tree from training data"""
        self.tree = self._build_tree(X, y, depth=0)
        print("✓ Decision Tree built successfully")
    
    def _build_tree(self, X: List[List[float]], y: List[str], depth: int) -> DecisionNode:
        """Recursively build the tree"""
        if depth >= self.max_depth or len(set(y)) == 1:
            return DecisionNode(value=max(set(y), key=y.count))
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Find best split
        for feature_idx in range(len(X[0])):
            values = sorted(set([x[feature_idx] for x in X]))
            
            for threshold in values:
                left_indices = [i for i in range(len(X)) if X[i][feature_idx] <= threshold]
                right_indices = [i for i in range(len(X)) if X[i][feature_idx] > threshold]
                
                if not left_indices or not right_indices:
                    continue
                
                gain = self._information_gain([y[i] for i in left_indices], 
                                             [y[i] for i in right_indices], y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            return DecisionNode(value=max(set(y), key=y.count))
        
        left_indices = [i for i in range(len(X)) if X[i][best_feature] <= best_threshold]
        right_indices = [i for i in range(len(X)) if X[i][best_feature] > best_threshold]
        
        left_subtree = self._build_tree([X[i] for i in left_indices], 
                                       [y[i] for i in left_indices], depth + 1)
        right_subtree = self._build_tree([X[i] for i in right_indices], 
                                        [y[i] for i in right_indices], depth + 1)
        
        return DecisionNode(feature=best_feature, threshold=best_threshold, 
                           left=left_subtree, right=right_subtree)
    
    def _information_gain(self, left_y: List[str], right_y: List[str], parent_y: List[str]) -> float:
        """Calculate information gain for a split"""
        parent_entropy = self._entropy(parent_y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)
        
        n = len(parent_y)
        gain = parent_entropy - (len(left_y)/n * left_entropy + len(right_y)/n * right_entropy)
        return gain
    
    def _entropy(self, y: List[str]) -> float:
        """Calculate entropy"""
        if not y:
            return 0
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        
        entropy = 0
        for count in counts.values():
            p = count / len(y)
            if p > 0:
                entropy -= p * (p ** 0.5)  # Simplified entropy
        return entropy
    
    def predict(self, X: List[List[float]]) -> List[str]:
        """Make predictions"""
        predictions = []
        for x in X:
            predictions.append(self._predict_sample(x, self.tree))
        return predictions
    
    def _predict_sample(self, x: List[float], node: DecisionNode) -> str:
        """Predict single sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)


class KNearestNeighbors:
    """K-Nearest Neighbors Classifier"""
    
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: List[List[float]], y: List[str]):
        """Store training data"""
        self.X_train = X
        self.y_train = y
        print("✓ KNN model fitted successfully")
    
    def predict(self, X: List[List[float]]) -> List[str]:
        """Make predictions using KNN"""
        predictions = []
        for x in X:
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._euclidean_distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            # Get k nearest neighbors
            distances.sort()
            k_nearest = distances[:self.k]
            
            # Vote for majority class
            labels = [label for _, label in k_nearest]
            prediction = max(set(labels), key=labels.count)
            predictions.append(prediction)
        
        return predictions
    
    def _euclidean_distance(self, x1: List[float], x2: List[float]) -> float:
        """Calculate Euclidean distance"""
        return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5


def main():
    """Main function demonstrating AI algorithms"""
    print("=" * 50)
    print("Java-style AI Working Example")
    print("=" * 50)
    
    # Sample data: [feature1, feature2, feature3] -> label
    X_train = [
        [2.0, 3.0, 1.0],
        [1.5, 2.5, 1.2],
        [7.0, 8.0, 2.0],
        [8.0, 7.5, 2.1],
        [3.0, 2.5, 1.1],
        [7.5, 8.5, 1.9]
    ]
    y_train = ["ClassA", "ClassA", "ClassB", "ClassB", "ClassA", "ClassB"]
    
    X_test = [
        [2.5, 3.5, 1.0],
        [7.5, 8.0, 2.0],
        [1.0, 2.0, 1.5]
    ]
    
    # Test 1: Decision Tree Classifier
    print("\n1. Decision Tree Classifier:")
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_train, y_train)
    dt_predictions = dt.predict(X_test)
    
    print("Predictions:")
    for i, pred in enumerate(dt_predictions):
        print(f"  Sample {i+1}: {X_test[i]} -> {pred}")
    
    # Test 2: K-Nearest Neighbors
    print("\n2. K-Nearest Neighbors (K=3):")
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    
    print("Predictions:")
    for i, pred in enumerate(knn_predictions):
        print(f"  Sample {i+1}: {X_test[i]} -> {pred}")
    
    print("\n" + "=" * 50)
    print("AI Classification Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
