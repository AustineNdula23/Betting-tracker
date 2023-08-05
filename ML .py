import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data: bets placed and outcomes (1 for win, 0 for loss)
bets = {'bet_amount': [10, 20, 30, 15, 25],
        'outcome': [1, 0, 1, 1, 0]}

# Convert data to numpy arrays
X = np.array(bets['bet_amount']).reshape(-1, 1)
y = np.array(bets['outcome'])

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Function to predict outcome based on bet amount
def predict_outcome(bet_amount):
    predicted_outcome = model.predict(np.array([[bet_amount]]))
    return predicted_outcome

# Test the model
test_bet = 40
predicted_result = predict_outcome(test_bet)
if predicted_result > 0.5:
    print(f"If you bet ${test_bet}, the model predicts a win.")
else:
    print(f"If you bet ${test_bet}, the model predicts a loss.")
