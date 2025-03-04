import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def main():
    # Load the trained GridSearchCV Decision Tree model
    model_path = r"C:\Users\abhin\Desktop\machine_learning\decision_tree(house_price_prediction,flower_classification)\trained_decissiontree_classifier.pkl"
    model = joblib.load(model_path)

    # Use the best estimator from GridSearchCV for visualization
    best_model = model.best_estimator_

    # Define feature names (for the Iris dataset)
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Visualize the decision tree using the best estimator
    plt.figure(figsize=(20, 10))
    plot_tree(
        best_model,
        feature_names=feature_names,
        class_names=["setosa", "versicolor", "virginica"],
        filled=True,
        rounded=True,
        fontsize=12
    )
    plt.title("Decision Tree Visualization")
    plt.show()

    # Get user input for each feature
    print("Enter values for the following features:")
    user_input = []
    for feature in feature_names:
        try:
            value = float(input(f"{feature}: "))
        except ValueError:
            print(f"Invalid input for {feature}. Please enter a numeric value.")
            return
        user_input.append(value)

    # Convert input to NumPy array and reshape for prediction
    user_input = np.array(user_input).reshape(1, -1)

    # Make prediction using the GridSearchCV object (it will use the best estimator)
    prediction = model.predict(user_input)

    # Class mapping (Iris dataset species)
    class_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
    print(f"Predicted Species: {class_mapping[prediction[0]]}")

if __name__ == '__main__':
    main()
