import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_recall_fscore_support, precision_score, recall_score, f1_score
)
from sklearn.decomposition import PCA

# 1) Cargar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 2) Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3) Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# 4) Métricas
acc = accuracy_score(y_test, y_pred)
prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
    y_test, y_pred, labels=[0,1,2], zero_division=0
)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)

print("=== Gaussian Naive Bayes (Iris) ===")
print(f"Accuracy: {acc:.4f}\n")
print(f"Precision (macro):   {precision_score(y_test, y_pred, average='macro',   zero_division=0):.3f}")
print(f"Recall (macro):      {recall_score(y_test, y_pred,    average='macro',   zero_division=0):.3f}")
print(f"F1-Score (macro):    {f1_score(y_test, y_pred,         average='macro',   zero_division=0):.3f}")
print(f"Precision (weighted):{precision_score(y_test, y_pred, average='weighted',zero_division=0):.3f}")
print(f"Recall (weighted):   {recall_score(y_test, y_pred,    average='weighted',zero_division=0):.3f}")
print(f"F1-Score (weighted): {f1_score(y_test, y_pred,         average='weighted',zero_division=0):.3f}\n")

print("Reporte de clasificación:\n",
      classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# 5) Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
im = ax.imshow(cm, interpolation='nearest')
ax.set_title("Matriz de Confusión - GaussianNB")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

tick_marks = np.arange(len(class_names))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_ylabel('Etiqueta real')
ax.set_xlabel('Etiqueta predicha')

th = cm.max() / 2.0 if cm.max() > 0 else 0.5
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > th else "black")
plt.tight_layout()
plt.show()

# 6) Gráfica de Métricas por clase + macro
labels = list(class_names) + ["macro"]
prec_vals = list(prec_cls) + [prec_macro]
rec_vals  = list(rec_cls)  + [rec_macro]
f1_vals   = list(f1_cls)   + [f1_macro]

x = np.arange(len(labels))
width = 0.25

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.bar(x - width, prec_vals, width, label='Precision')
ax.bar(x,         rec_vals,  width, label='Recall')
ax.bar(x + width, f1_vals,   width, label='F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Métricas por clase y macro (GaussianNB)")
ax.legend()
plt.tight_layout()
plt.show()

# 7) Graph pre-process data: PCA 2D con etiquetas reales
pca_full = PCA(n_components=2, random_state=42)
X_pca = pca_full.fit_transform(X)

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
for cls in np.unique(y):
    ax.scatter(X_pca[y==cls, 0], X_pca[y==cls, 1], label=class_names[cls], s=25)
ax.set_title("Pre-process (PCA 2D) — Etiquetas reales")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
plt.tight_layout()
plt.show()

# 8) Graph process data: regiones de decisión (visualización)
pca_viz = PCA(n_components=2, random_state=42)
X_train_pca = pca_viz.fit_transform(X_train)
X_test_pca  = pca_viz.transform(X_test)

gnb_viz = GaussianNB()
gnb_viz.fit(X_train_pca, y_train)

h = 0.02
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = gnb_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.contourf(xx, yy, Z, alpha=0.25)

ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=15, alpha=0.6, label="Train")
correct = y_test == gnb.predict(X_test)
ax.scatter(X_test_pca[correct, 0], X_test_pca[correct, 1], s=35, marker='o', label="Test - correcto")
ax.scatter(X_test_pca[~correct, 0], X_test_pca[~correct, 1], s=60, marker='x', label="Test - error")

ax.set_title("Process (PCA 2D) — Regiones de decisión (GaussianNB)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(loc="best")
plt.tight_layout()
plt.show()

# 9) Dispersión simple en pares de features
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
for cls in np.unique(y):
    ax.scatter(X[y==cls, 0], X[y==cls, 1], label=class_names[cls], s=20)
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_title("Sepal Length vs Sepal Width")
ax.legend()
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
for cls in np.unique(y):
    ax.scatter(X[y==cls, 2], X[y==cls, 3], label=class_names[cls], s=20)
ax.set_xlabel(feature_names[2])
ax.set_ylabel(feature_names[3])
ax.set_title("Petal Length vs Petal Width")
ax.legend()
plt.tight_layout()
plt.show()
