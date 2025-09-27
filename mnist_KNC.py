# ============================================
# MNIST + KNeighborsClassifier (784D)
# Entrenamiento directo con los 784 pixeles
# Visualización en 2D via PCA (solo para dibujar)
# Métricas: Accuracy, Precision, Recall, F1-Score
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.decomposition import PCA
import matplotlib as mpl

# -----------------------------
# 0) Estética global de gráficas
# -----------------------------
mpl.rcParams.update({
    "figure.dpi": 140,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.facecolor": "white"  # útil en Colab
})

# -----------------------------
# 1) Cargar MNIST
# -----------------------------
from tensorflow.keras.datasets import mnist
np.random.seed(42)

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# -----------------------------
# 2) Pre-proceso básico
# -----------------------------
X_train = (X_train_raw.astype("float32")/255.0).reshape(-1, 28*28)  # 784D
X_test  = (X_test_raw.astype("float32")/255.0).reshape(-1, 28*28)
y_train = y_train_raw.astype(int)
y_test  = y_test_raw.astype(int)

# Subconjunto de entrenamiento para acelerar
TRAIN_SIZE = 20000
idx_tr = np.random.choice(len(X_train), size=TRAIN_SIZE, replace=False)
Xtr_sub, ytr_sub = X_train[idx_tr], y_train[idx_tr]

# -----------------------------
# 3) Entrenar KNN directamente en 784D
# -----------------------------
knn_main = KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)
knn_main.fit(Xtr_sub, ytr_sub)
y_pred = knn_main.predict(X_test)

# -----------------------------
# 4) Métricas + gráfica de barras (estética mejorada)
# -----------------------------
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("Metrics to show")
print(f"- Accuracy:  {acc:.4f}")
print(f"- Precision: {prec:.4f}")
print(f"- Recall:    {rec:.4f}")
print(f"- F1-Score:  {f1:.4f}")

plt.figure(figsize=(6.2, 4.2))
labels = ["Accuracy","Precision","Recall","F1-Score"]
vals = [acc,prec,rec,f1]
bars = plt.bar(labels, vals)
plt.ylim(0, 1.02)
plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
             ha="center", va="bottom", fontsize=10)
plt.ylabel("Score")
plt.title("KNN en MNIST (784D) — Métricas")
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Matriz de confusión (normalizada + conteos)
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=np.arange(10))
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6.8, 5.6))
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
ax.set_title("Matriz de confusión — Normalizada por clase")
ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
ax.set_xticks(np.arange(10)); ax.set_yticks(np.arange(10))
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.grid(False)

thr = cm_norm.max()/2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        pct = cm_norm[i, j]*100
        txt = f"{pct:.1f}%\n({cm[i,j]})"
        ax.text(j, i, txt,
                ha="center", va="center",
                fontsize=8.7,
                color="white" if cm_norm[i,j] > thr else "black")
plt.tight_layout()
plt.show()

# -----------------------------
# 6) PCA 2D SOLO para visualización
# -----------------------------
pca2 = PCA(n_components=2, random_state=42)
Xtr2_full = pca2.fit_transform(Xtr_sub)   # train-sub reducido a 2D (solo para graficar)
Xte2      = pca2.transform(X_test)

# Scatter de entrenamiento (etiquetas reales), submuestreo para legibilidad
VIS_N = 5000
idx_vis = np.random.choice(len(Xtr2_full), size=min(VIS_N, len(Xtr2_full)), replace=False)

plt.figure(figsize=(7.5, 6.2))
sc = plt.scatter(Xtr2_full[idx_vis,0], Xtr2_full[idx_vis,1],
                 c=ytr_sub[idx_vis], s=10, alpha=0.75, cmap="tab10", linewidths=0)
plt.title("PCA 2D (Train Sub) — Etiquetas reales")
plt.xlabel("PC1"); plt.ylabel("PC2")
cbar = plt.colorbar(sc, ticks=range(10)); cbar.set_label("Etiqueta real")
plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------
# 7) Fronteras de decisión (KNN en 2D) + predicciones
#    Mejora visual: malla densa + suavizado + puntos de test submuestreados
# -----------------------------
knn_pca2d = KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)
knn_pca2d.fit(Xtr2_full, ytr_sub)

# Malla para fronteras (ajusta RES si quieres más/menos detalle)
pad = 2.0
x_min, x_max = Xtr2_full[:,0].min()-pad, Xtr2_full[:,0].max()+pad
y_min, y_max = Xtr2_full[:,1].min()-pad, Xtr2_full[:,1].max()+pad

RES = 500  # resolución de la malla (más alto = más suave)
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, RES),
    np.linspace(y_min, y_max, RES)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = knn_pca2d.predict(grid).reshape(xx.shape)

# Predicciones de test en 2D (submuestreo para no saturar)
N_TE_VIS = 8000
idx_te_vis = np.random.choice(len(Xte2), size=min(N_TE_VIS, len(Xte2)), replace=False)
Xte2_vis = Xte2[idx_te_vis]
y_pred_pca2_vis = knn_pca2d.predict(Xte2_vis)

# Dibujo de fronteras (suavizado) + puntos
plt.figure(figsize=(8.2, 6.6))

# 1) Fondo con regiones de decisión
#    pcolormesh suele verse más “relleno” que contourf para este tipo de mapas
plt.pcolormesh(xx, yy, Z, shading="auto", alpha=0.20, cmap="tab10")

# 2) Contornos finos para separar regiones
CS = plt.contour(xx, yy, Z, colors="k", linewidths=0.35, alpha=0.55)

# 3) Puntos de test (predicciones) por encima
sc2 = plt.scatter(Xte2_vis[:,0], Xte2_vis[:,1],
                  c=y_pred_pca2_vis, s=9, alpha=0.85,
                  cmap="tab10", edgecolor="none")

plt.title("PCA 2D — Fronteras KNN y predicciones (subset de test)")
plt.xlabel("PC1"); plt.ylabel("PC2")
cbar2 = plt.colorbar(sc2, ticks=range(10)); cbar2.set_label("Etiqueta predicha")
plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------
# (Opcional) 8) Comparar decisión PCA2D vs. real 784D
# -----------------------------
# Nota: El clasificador "bueno" es el de 784D; el de 2D es solo para dibujar.
# Si quieres, puedes imprimir también métricas del modelo 2D:
# y_pred_pca2_full = knn_pca2d.predict(Xte2)
# print("Accuracy (PCA2D):", accuracy_score(y_test, y_pred_pca2_full))
