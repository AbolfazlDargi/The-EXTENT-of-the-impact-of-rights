import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

years_of_experience = np.array([0,1, 2, 3, 4, 6, 8,10]).reshape(-1,1)
years_salieser = np.array([30000, 40000, 55000, 60000, 70000, 80000, 85000, 87000])

# plt.plot(years_of_experience, years_salieser, '*')
# plt.show()

model = DecisionTreeRegressor(max_depth=4)
model.fit(years_of_experience, years_salieser)

y_predict = model.predict(years_of_experience)

plt.plot(years_of_experience, years_salieser, '*',)
plt.plot(years_of_experience, y_predict)
plt.title('Decision Tree Regrssor (Correct Model)')
plt.
plt.show()


