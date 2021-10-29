# SGD-Linear-Regressor
Iterative stochastic gradient descent (SGD) linear regressor with regularization

* Dataset: Kaggle “Graduate Admission 2” https://www.kaggle.com/mohansacharya/. The dataset contains a number of parameters: 
  1. GRE Scores (out of 340)
  2. TOEFL Scores (out of 120)
  3. University Rating (out of 5) 
  4. Statement of Purpose (out of 5)
  5. Letter of Recommendation Strength (out of 5)
  6. Undergraduate GPA (out of 10)
  7. Research Experience (either 0 or 1)
  8. Chance of Admit (ranging from 0 to 1)
* SGD solver supports 2D grid search with one dimension being the learning rate α and the other dimension being the regularization weight λ.
* The loss (error) in regression is defined as the mean squared error (MSE) between the ground truth values and the regression values.
