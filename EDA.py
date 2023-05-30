# Импортируем необходимые библиотеки
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем дата-сет
df = pd.read_csv('diabetes.csv')

# Проводим общую оценку данных
df.info()
df.describe()

# Строим корреляционную матрицу
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
