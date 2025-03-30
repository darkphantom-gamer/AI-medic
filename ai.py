from sklearn.linear_model import LinearRegression
import numpy as np
import 

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

print(f"Prediction for 6: {model.predict([[6]])[0]}")
print("complete")