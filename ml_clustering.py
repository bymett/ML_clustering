import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns

data = []
age = []
sex = []
prof = []
inc = []
c = 0
df = pd.read_csv('clustering_1_3008.csv')
df = df[~(df['sex'].isnull() | df['age'].isnull() | df['profession'].isnull() | df['income'].isnull())]
df = df.reset_index(drop=True)
print(df)

plt.scatter(df['age'], df['income'])
plt.xlabel('возраст')
plt.ylabel('заработок')
plt.title('Диаграмма рассеяния')
plt.show()

plt.boxplot(df['age'])
plt.title('Диаграмма Box-and-Whisker1')
plt.show()

plt.boxplot(df['income'])
plt.title('Диаграмма Box-and-Whisker2')
plt.show()

sns.countplot(data=df, x='sex')
plt.xlabel('пол')
plt.ylabel('Частота')
plt.title('Гистограмма категориального признака')
plt.xticks(rotation=45)
plt.show()

sns.countplot(data=df, x='profession')
plt.xlabel('профессия')
plt.ylabel('Частота')
plt.title('Гистограмма категориального признака')
plt.xticks(rotation=45)
plt.show()

from sklearn.preprocessing import OneHotEncoder


if 'sex_male' in df.columns:
    df = df.drop(columns=['sex_male'])
else:
    encoder = OneHotEncoder(drop='first')
    X = df['sex'].values.reshape(-1, 1)
    encoded_data = encoder.fit_transform(X).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['sex']))
    df = pd.concat([df, encoded_df], axis=1)

if 'a' in df.columns:
    df = df.drop(columns=['sex_male'])
else:
    encoder = OneHotEncoder(drop='first', sparse=False)

# Примените кодирование к столбцу 'color' и добавьте результат в DataFrame
    encoded_data = encoder.fit_transform(df[['profession']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['profession']))

# Добавьте закодированные столбцы к исходному DataFrame
    df = pd.concat([df, encoded_df], axis=1)

# Выведите новый DataFrame

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Корреляционная матрица признаков')
plt.show()
print(df)

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

X = df[['age', 'income', 'sex_male', 'profession_unemployed', 'profession_worker']]
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
n_components = 2
pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train)
df_pca = pd.DataFrame(data=X_train_pca, columns=['ГК1', 'ГК2'])

print(df_pca)

# Получите матрицу нагрузок из объекта PCA
loadings_matrix = pca.components_

# Индекс 1-ой главной компоненты в матрице нагрузок
index_of_first_principal_component = 0
index_of_second_principal_component = 1

# Получите коэффициенты нагрузок для 1-ой главной компоненты
loadings_for_first_component = loadings_matrix[index_of_first_principal_component, :]
loadings_for_second_component = loadings_matrix[index_of_second_principal_component, :]
# Создайте словарь, связывающий признаки с их коэффициентами нагрузок
feature_loadings1 = dict(zip(X.columns, loadings_for_first_component))
feature_loadings2 = dict(zip(X.columns, loadings_for_second_component))

# Сортируйте признаки по абсолютному значению коэффициентов нагрузок
sorted_features1 = sorted(feature_loadings1.items(), key=lambda x: abs(x[1]), reverse=True)
sorted_features2 = sorted(feature_loadings2.items(), key=lambda x: abs(x[1]), reverse=True)

# Печатайте признак с наибольшей корреляцией с 1-ой главной компонентой
print(f"Признак с наибольшей корреляцией с 1-ой главной компонентой: {sorted_features1[0]}")
print(f"Признак с наибольшей корреляцией с 2-ой главной компонентой: {sorted_features2[0]}")

print(loadings_matrix)

plt.scatter(df_pca['ГК1'], df_pca['ГК2'])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Данные после факторизации методом PCA (Главная Компонента 1 и Главная Компонента 2)
X_pca = df_pca[['ГК1', 'ГК2']]

# Создайте экземпляр модели KMeans с 3 кластерами
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Выполните кластеризацию на данных после PCA
cluster_labels = kmeans.fit_predict(X_pca)

# Рассчитайте коэффициент силуэта для оценки качества кластеризации
silhouette_avg = silhouette_score(X_pca, cluster_labels)
print("Коэффициент силуэта:", silhouette_avg)

import matplotlib.pyplot as plt

# Добавляем метки кластеров в исходный DataFrame
df_pca['Кластер'] = cluster_labels

# Построение графика с использованием цветов для каждого кластера
plt.figure(figsize=(8, 6))

# Определение цветов для кластеров
colors = ['red', 'blue', 'green']

# Построение точек для каждого кластера и вычисление средних значений главных компонент
for cluster_num, color in enumerate(colors):
    cluster_data = df_pca[df_pca['Кластер'] == cluster_num]
    plt.scatter(cluster_data['ГК1'], cluster_data['ГК2'], color=color, label=f'Кластер {cluster_num}')

    # Вычисление средних значений главных компонент для текущего кластера
    mean_pc1 = round(cluster_data['ГК1'].mean())
    mean_pc2 = round(cluster_data['ГК2'].mean())

    # Вывод средних значений рядом с кластером
    plt.annotate(f'Mean PC1: {mean_pc1}\nMean PC2: {mean_pc2}',
                 xy=(mean_pc1, mean_pc2), xytext=(mean_pc1 + 1, mean_pc2 + 1),
                 arrowprops=dict(arrowstyle='->'), color=color)

# Добавление подписей осей и заголовка
plt.xlabel('Главная Компонента 1')
plt.ylabel('Главная Компонента 2')
plt.title('Кластеризация данных после PCA')

# Отображение легенды
plt.legend(loc='upper right')

# Отображение графика
plt.grid(True)
plt.show()