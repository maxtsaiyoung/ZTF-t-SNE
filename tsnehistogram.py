import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === Step 1: Load Data ===
file_path = "/data/Bulk_LS/LS_Result/0760_14_3.result"
data = np.loadtxt(file_path)

# === Step 2: Select Middle 8 Features (skip first 3 and last 2 columns)
colnames = ["RA", "Dec", "PS_ID", "best_period", "sig_best", "sig_half", "sig_twice",
            "mad_cal", "var_cal", "skew", "kurtosis", "ref_flag", "ref_flux"]

feature_indices = list(range(3, 11))  # Columns 3 through 10 (8 features)
feature_names = [colnames[i] for i in feature_indices]
X = data[:, feature_indices]

# === Step 3: Plot Histograms of Selected Features ===
for i in range(X.shape[1]):
    plt.figure()
    plt.hist(X[:, i], bins=50, alpha=0.7)
    plt.title(f"Histogram of {feature_names[i]}")
    plt.xlabel(feature_names[i])
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("histogram_" + feature_names[i] + ".png", dpi=300)
    print(f"Histogram for {feature_names[i]} saved as 'histogram_{feature_names[i]}.png'.")

# === Step 4: Apply t-SNE ===
X_embed = TSNE(n_components=2, perplexity=100, n_iter=1000).fit_transform(X)
np.save('tsne_embedding_0760_14_3_mid8.npy', X_embed)  # Save embedding

# === Step 5: Highlight Known White Dwarf ===
RA_target = 234.883995
Dec_target = 50.46077304
tolerance = 3e-5

mask = (np.abs(data[:, 0] - RA_target) < tolerance) & (np.abs(data[:, 1] - Dec_target) < tolerance)

# === Step 6: Plot t-SNE Scatter Plot ===
plt.figure(figsize=(8, 6))
plt.scatter(X_embed[:, 0], X_embed[:, 1], s=5, alpha=0.5, label='Stars')
if np.any(mask):
    plt.scatter(X_embed[mask, 0], X_embed[mask, 1], color='red', s=30, marker='x', label='Known white dwarf')
else:
    print("Kevin's star not found — check tolerance or input data")

plt.title("t-SNE on Middle 8 LS Variability Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid()
plt.savefig("tsne_plot_mid8_0760_14_3.png", dpi=300)

# === Step 7: Apply 3-Component t-SNE and Plot ===
X_embed_3d = TSNE(n_components=3, perplexity=100, n_iter=1000).fit_transform(X)
np.save('tsne_embedding_0760_14_3_mid8_3d.npy', X_embed_3d)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_embed_3d[:, 0], X_embed_3d[:, 1], X_embed_3d[:, 2], s=5, alpha=0.5, label='Stars')
if np.any(mask):
    ax.scatter(X_embed_3d[mask, 0], X_embed_3d[mask, 1], X_embed_3d[mask, 2], color='red', s=30, marker='x', label='Known white dwarf')
else:
    print("Kevin's star not found in 3D t-SNE — check tolerance or input data")
ax.set_title("3D t-SNE on Middle 8 LS Variability Features")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_zlabel("t-SNE Component 3")
ax.legend()
plt.savefig("tsne_plot_mid8_0760_14_3_3d.png", dpi=300)


