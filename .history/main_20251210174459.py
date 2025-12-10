import matplotlib.pyplot as plt
import sklearn.tree as tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier

years_of_experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
years_salieser = np.array([45000, 50000, 60000, 65000, 70000, 80000, 85000, 90000, 95000, 100000])

plt.plot(years_of_experience, years_salieser, '*')
plt.show()

model = DecisionTreeClassifier(max_depth=2)
model.fit(years_of_experience, years_salieser)
y_predict = model.predict(years_of_experience,)


