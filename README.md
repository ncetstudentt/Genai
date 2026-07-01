# GenAI - AI Learning Repository

Welcome to **GenAI**, a comprehensive repository showcasing AI and Machine Learning implementations. This repo contains practical examples of how AI algorithms work, written in both Python and Java-style code.

## 📁 Repository Structure

```
Genai/
├── Python_ai/
│   └── script.py          # Python AI implementations
├── java_ai/
│   └── script.py          # Java-style AI implementations
└── README.md              # This file
```

## 🤖 What's Inside?

### 1. Python AI (`Python_ai/script.py`)

A Python implementation demonstrating fundamental AI concepts:

#### **SimpleNeuralNetwork Class**
- Basic 3-layer neural network architecture
- Forward propagation through hidden and output layers
- Sigmoid activation function for non-linear transformations
- Prediction capabilities for classification tasks

**Features:**
- Customizable input, hidden, and output layer sizes
- Random weight initialization
- Forward pass computation with activation functions

**Example:**
```python
nn = SimpleNeuralNetwork(input_size=3, hidden_size=4, output_size=2)
prediction = nn.predict([0.1, 0.2, 0.3])  # Returns predicted class (0 or 1)
```

#### **SimpleAI Class**
- Pattern recognition and learning system
- Knowledge base for storing learned patterns
- Similarity-based prediction with confidence scores

**Features:**
- Learn from training data with `learn()` method
- Recognize patterns in text using pattern matching
- Make predictions with confidence scores
- Handles unknown patterns gracefully

**Example:**
```python
ai = SimpleAI()
ai.learn("hello friend", "greeting")
prediction, confidence = ai.predict("hello world")  # Returns ("greeting", confidence_score)
```

**How it works:**
1. Train the AI with labeled examples
2. The system stores patterns in a knowledge base
3. For predictions, it finds matching patterns and calculates similarity scores
4. Returns the best match with confidence level

---

### 2. Java-Style AI (`java_ai/script.py`)

Object-oriented AI implementations featuring classification algorithms:

#### **Decision Tree Classifier**
- Recursive tree building algorithm
- Information gain calculation for optimal splits
- Entropy-based node purity measurement
- Configurable maximum depth

**Features:**
- Automatic feature and threshold selection
- Builds optimal decision trees from training data
- Predicts class labels for new samples
- Prevents overfitting with depth limits

**Example:**
```python
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
```

**How it works:**
1. Recursively splits data based on features that maximize information gain
2. Each split reduces entropy (disorder) in the data
3. Builds a tree structure where leaves represent class predictions
4. Makes predictions by following the tree path for new data

#### **K-Nearest Neighbors (KNN)**
- Instance-based learning algorithm
- Euclidean distance calculation
- k-nearest neighbor voting mechanism
- Configurable k parameter

**Features:**
- Simple yet effective classification
- k neighbors vote on the predicted class
- Uses Euclidean distance to find similar samples
- No explicit training phase (lazy learning)

**Example:**
```python
knn = KNearestNeighbors(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

**How it works:**
1. Calculates distance from test sample to all training samples
2. Selects k closest neighbors
3. Majority vote among k neighbors determines prediction
4. Simple, intuitive, and effective for many datasets

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- No external dependencies required (uses only built-in Python libraries)

### Running the Examples

#### Python AI Examples:
```bash
cd Python_ai
python script.py
```

**Output includes:**
- Pattern recognition predictions on test inputs
- Confidence scores for each prediction
- Neural network forward pass results

#### Java-Style AI Examples:
```bash
cd java_ai
python script.py
```

**Output includes:**
- Decision tree classifications
- KNN predictions with majority voting
- Performance on test datasets

---

## 📊 Algorithm Comparison

| Algorithm | Type | Complexity | Best For |
|-----------|------|-----------|----------|
| **Neural Network** | Deep Learning | O(n × m) | Complex patterns, non-linear data |
| **SimpleAI** | Pattern Matching | O(n) | Text/pattern recognition |
| **Decision Tree** | Supervised Learning | O(log n) | Interpretable decisions, small datasets |
| **KNN** | Instance-based | O(n × d) | Simple datasets, fast learning |

---

## 🎯 Use Cases

### Pattern Recognition (Python AI)
- Text classification
- Sentiment analysis
- Email spam detection
- Content categorization

### Classification (Java-Style AI)
- Decision making systems
- Data classification
- Predictive modeling
- Feature-based categorization

---

## 📝 Code Examples

### Example 1: Training SimpleAI for Greetings

```python
from Python_ai.script import SimpleAI

# Create and train AI
ai = SimpleAI()
ai.learn("hello friend", "greeting")
ai.learn("hi there", "greeting")
ai.learn("goodbye see you", "farewell")
ai.learn("bye take care", "farewell")

# Make predictions
prediction, confidence = ai.predict("hello world")
print(f"Predicted: {prediction}, Confidence: {confidence:.2f}")
```

### Example 2: Decision Tree Classification

```python
from java_ai.script import DecisionTreeClassifier

# Training data
X_train = [[2.0, 3.0, 1.0], [7.0, 8.0, 2.0], ...]
y_train = ["ClassA", "ClassB", ...]

# Train and predict
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
predictions = dt.predict([[3.0, 2.5, 1.1]])
```

---

## 🔧 How AI Works (Simplified)

### Training Process:
1. **Data Collection**: Gather labeled training examples
2. **Feature Extraction**: Identify important characteristics
3. **Algorithm Selection**: Choose appropriate ML algorithm
4. **Model Training**: Learn patterns from data
5. **Validation**: Test on unseen data

### Prediction Process:
1. **Input**: Receive new data sample
2. **Processing**: Extract features from input
3. **Comparison**: Compare against learned patterns
4. **Decision**: Make prediction based on similarities
5. **Output**: Return predicted label with confidence

---

## 📚 Learning Resources

- **Neural Networks**: Understanding backpropagation and activation functions
- **Decision Trees**: Information theory and entropy calculation
- **KNN**: Distance metrics and lazy learning
- **Pattern Recognition**: Text processing and similarity measures

---

## 🤝 Contributing

Feel free to enhance these implementations:
- Add more algorithms (Random Forest, SVM, etc.)
- Improve accuracy metrics
- Add visualization capabilities
- Expand documentation with more examples

---

## 📄 License

This repository is open for educational purposes.

---

## ✨ Key Features

✅ **Pure Python** - No heavy dependencies  
✅ **Well-Documented** - Clear comments and examples  
✅ **Educational** - Learn AI concepts from scratch  
✅ **Practical** - Runnable examples included  
✅ **Extensible** - Easy to modify and enhance  

---

## 🎓 Learning Outcomes

After exploring this repository, you'll understand:
- ✓ How neural networks make predictions
- ✓ Decision tree construction and splitting
- ✓ K-Nearest Neighbors algorithm
- ✓ Pattern recognition techniques
- ✓ Machine learning workflow
- ✓ Information theory basics (entropy, information gain)

---

**Happy Learning! 🚀**

*For questions or suggestions, feel free to contribute or open an issue.*
