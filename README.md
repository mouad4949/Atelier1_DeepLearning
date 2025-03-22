# PyTorch Classification and Regression Lab

## Objective
The main goal of this lab is to gain hands-on experience with the PyTorch library for handling Classification and Regression tasks using Deep Neural Networks (DNN) and Multi-Layer Perceptrons (MLP).

---

## Woork to Do

### Part 1: Regression
**Dataset:** [NYSE Stock Market Data](https://www.kaggle.com/datasets/dgawlik/nyse)

1. **Exploratory Data Analysis (EDA):**
   - Apply data exploration techniques to understand and visualize the dataset.
   - Identify missing values, outliers, and correlations between features.

2. **Model Development:**
   - Design a Deep Neural Network (DNN) architecture using PyTorch to perform regression.

3. **Hyperparameter Tuning:**
   - Use the `GridSearchCV` tool from scikit-learn to find the optimal hyperparameters (learning rate, optimizers, number of epochs, model architecture, etc.) for the best performance.

4. **Model Performance Visualization:**
   - Plot the Loss vs. Epochs and Accuracy vs. Epochs graphs for both training and test data.
   - Interpret the results and discuss any observed trends.

5. **Regularization Techniques:**
   - Implement regularization methods (such as dropout, weight decay, batch normalization) to prevent overfitting.
   - Compare the performance of the regularized model with the initial model.

---

### Part 2: Multi-Class Classification
**Dataset:** [Predictive Maintenance Classification Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

1. **Preprocessing:**
   - Apply preprocessing techniques to clean and standardize/normalize the dataset.

2. **Exploratory Data Analysis (EDA):**
   - Visualize and analyze the dataset to understand its structure and patterns.

3. **Data Augmentation:**
   - Apply data augmentation techniques to balance the dataset and improve model generalization.

4. **Model Development:**
   - Design a Deep Neural Network (DNN) architecture using PyTorch for multi-class classification.

5. **Hyperparameter Tuning:**
   - Use `GridSearchCV` from scikit-learn to determine the best hyperparameters for model training.

6. **Model Performance Visualization:**
   - Plot the Loss vs. Epochs and Accuracy vs. Epochs graphs for both training and test data.
   - Provide an interpretation of the model's learning process.

7. **Evaluation Metrics:**
   - Compute accuracy, sensitivity, F1-score, and other relevant metrics for both training and test sets.

8. **Regularization Techniques:**
   - Implement different regularization methods and compare the results with the initial model.

---

## Tools & Libraries
- Python
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib / seaborn (for visualization)
- GridSearchCV (for hyperparameter tuning)

## How to Run
1. Install required libraries using:
   ```bash
   pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
   ```
2. Load the datasets from Kaggle.
3. Follow the structured steps for both regression and classification tasks.
4. Tune hyperparameters and apply regularization techniques.
5. Evaluate and visualize model performance.

---

## Expected Outcomes
- A well-trained regression model with optimized hyperparameters.
- A multi-class classification model with balanced dataset handling and high evaluation scores.
- Understanding of the impact of regularization techniques on model performance.
- Effective visualization of model learning and performance.

