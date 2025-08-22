from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Define the BasicDecisionTree class
class BasicDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < 2:
            leaf_value = self._most_common_label(y)
            return {'leaf': leaf_value}

        best_split = self._find_best_split(X, y)

        if best_split is None:
            leaf_value = self._most_common_label(y)
            return {'leaf': leaf_value}

        feature_index, threshold = best_split
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': feature_index,
                'threshold': threshold,
                'left': left_subtree,
                'right': right_subtree}

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_split = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                gini = self._calculate_gini(y[left_indices], y[right_indices])

                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, threshold)

        return best_split

    def _calculate_gini(self, left_y, right_y):
        n_left, n_right = len(left_y), len(right_y)
        n_total = n_left + n_right

        gini_left = 1.0
        if n_left > 0:
            _, counts_left = np.unique(left_y, return_counts=True)
            gini_left -= np.sum((counts_left / n_left)**2)

        gini_right = 1.0
        if n_right > 0:
            _, counts_right = np.unique(right_y, return_counts=True)
            gini_right -= np.sum((counts_right / n_right)**2)

        gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        return gini

    def _most_common_label(self, y):
        if len(y) == 0:
            return None
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if 'leaf' in node:
            return node['leaf']

        if x[node['feature_index']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

# Reverse size label mapping
reverse_size_mapping = {0: 'L', 1: 'M', 2: 'S', 3: 'XL', 4: 'XXL', 5: 'XXS', 6: 'XXXL'}

# Load the pre-trained model (ensure the correct path to the .pkl file)
with open('basic_decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the input values from the form
            weight = float(request.form['weight'])
            age = float(request.form['age'])
            height = float(request.form['height'])

            # Convert the features into numpy array and reshape
            features = np.array([weight, age, height]).reshape(1, -1)
            
            # Get the prediction
            prediction = model.predict(features)
            
            # Map the numeric prediction to the corresponding size
            predicted_size = reverse_size_mapping.get(prediction[0], 'Unknown')

            # Return the result to the user
            return render_template('result.html', prediction=predicted_size)
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
