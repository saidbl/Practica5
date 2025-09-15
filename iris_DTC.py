import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd

# Cargar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Crear y entrenar Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predicción
y_pred = dt.predict(X_test)

# Evaluación
acc = accuracy_score(y_test, y_pred)
print("=== Decision Tree sin escalado ===")
print("Accuracy:", acc)
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=class_names))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title("Matriz de Confusión - Decision Tree sin escalado")
plt.ylabel("Etiqueta Real")
plt.xlabel("Etiqueta Predicha")
plt.show()

# Gráficas de dispersión
plt.figure(figsize=(7,5))
for i, color in zip(range(3), ['red', 'green', 'blue']):
    plt.scatter(X[y==i, 0], X[y==i, 1], label=class_names[i], color=color)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Sepal Length vs Sepal Width - Decision Tree sin escalado")
plt.legend()
plt.show()

plt.figure(figsize=(7,5))
for i, color in zip(range(3), ['red', 'green', 'blue']):
    plt.scatter(X[y==i, 2], X[y==i, 3], label=class_names[i], color=color)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.title("Petal Length vs Petal Width - Decision Tree sin escalado")
plt.legend()
plt.show()