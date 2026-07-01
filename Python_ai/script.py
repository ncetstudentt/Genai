"""
Python AI Script - Simple Machine Learning Example
Demonstrates basic AI/ML concepts: data processing, model training, and predictions
"""

import random
from typing import List, Tuple

class SimpleNeuralNetwork:
    """A basic neural network for demonstration"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize network with random weights"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.w1 = [[random.random() for _ in range(hidden_size)] for _ in range(input_size)]
        self.w2 = [[random.random() for _ in range(output_size)] for _ in range(hidden_size)]
        self.b1 = [random.random() for _ in range(hidden_size)]
        self.b2 = [random.random() for _ in range(output_size)]
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        return 1 / (1 + 2.718 ** -x)
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward propagation through the network"""
        # Hidden layer
        hidden = []
        for j in range(self.hidden_size):
            z = self.b1[j]
            for i in range(self.input_size):
                z += inputs[i] * self.w1[i][j]
            hidden.append(self.sigmoid(z))
        
        # Output layer
        output = []
        for j in range(self.output_size):
            z = self.b2[j]
            for i in range(self.hidden_size):
                z += hidden[i] * self.w2[i][j]
            output.append(self.sigmoid(z))
        
        return output
    
    def predict(self, inputs: List[float]) -> int:
        """Make a prediction"""
        output = self.forward(inputs)
        return output.index(max(output))


class SimpleAI:
    """Simple AI for pattern recognition and decision making"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.training_data = []
    
    def learn(self, input_data: str, label: str):
        """Learn from data"""
        if label not in self.knowledge_base:
            self.knowledge_base[label] = []
        self.knowledge_base[label].append(input_data)
        self.training_data.append((input_data, label))
        print(f"✓ AI learned: '{input_data}' -> {label}")
    
    def recognize_pattern(self, text: str) -> str:
        """Recognize patterns in text"""
        text_lower = text.lower()
        
        # Pattern matching
        for label, patterns in self.knowledge_base.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    return label
        
        return "Unknown"
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict category with confidence"""
        best_match = "Unknown"
        best_score = 0.0
        
        for label, patterns in self.knowledge_base.items():
            for pattern in patterns:
                # Simple similarity score
                matches = sum(1 for word in pattern.split() if word.lower() in text.lower())
                score = matches / max(len(pattern.split()), 1)
                
                if score > best_score:
                    best_score = score
                    best_match = label
        
        return best_match, best_score


def main():
    """Main function demonstrating AI"""
    print("=" * 50)
    print("Python AI Working Example")
    print("=" * 50)
    
    # Example 1: Simple AI Learning
    print("\n1. Simple Pattern Recognition AI:")
    ai = SimpleAI()
    
    # Train the AI
    ai.learn("hello friend", "greeting")
    ai.learn("hi there", "greeting")
    ai.learn("goodbye see you", "farewell")
    ai.learn("bye take care", "farewell")
    ai.learn("what is AI", "question")
    ai.learn("how does machine learning work", "question")
    
    # Test predictions
    test_inputs = [
        "hello world",
        "goodbye friend",
        "what is deep learning",
        "random text"
    ]
    
    print("\nPredictions:")
    for test in test_inputs:
        prediction, confidence = ai.predict(test)
        print(f"  Input: '{test}' -> Prediction: {prediction} (confidence: {confidence:.2f})")
    
    # Example 2: Neural Network
    print("\n2. Simple Neural Network:")
    nn = SimpleNeuralNetwork(input_size=3, hidden_size=4, output_size=2)
    
    test_data = [
        [0.1, 0.2, 0.3],
        [0.5, 0.5, 0.5],
        [0.9, 0.8, 0.7]
    ]
    
    print("\nNeural Network Predictions:")
    for data in test_data:
        prediction = nn.predict(data)
        print(f"  Input: {data} -> Class: {prediction}")
    
    print("\n" + "=" * 50)
    print("AI Examples Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
