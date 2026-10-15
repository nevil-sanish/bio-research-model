import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Gene Expression Clustering (Topic 11, Algorithm M11 — K-Means)
**Student:** Nevil Mathew Sanish · **Roll:** 2025BCD0011 (R = 011)
**Paper:** Georgieva, *Clustering for Gene Expression Analysis*, ERIS 2022.

**Biological problem:** group tumour samples / genes by their expression profiles so that unknown
partitions (here, cancer types) can be discovered without labels.
**Dataset:** UCI *Gene Expression Cancer RNA-Seq* (TCGA PANCAN, 801 samples × 20,531 genes, 5 tumour types:
BRCA, COAD, KIRC, LUAD, PRAD). Labels are used **only for evaluation**.
**Algorithm:** K-Means (M11), k = 5, hyper-parameters tuned by 5-fold CV on training data. Evaluated on a held-out 20 % test set the model never saw.""")

code("""!pip -q install joblib
import os, io, json, zipfile, tarfile, urllib.request
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, confusion_matrix)
import joblib
SEED = 42
np.random.seed(SEED)""")

md("## 1. Download and load the dataset")
code("""URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
buf = io.BytesIO(urllib.request.urlopen(URL, timeout=600).read())
zf = zipfile.ZipFile(buf)
print(zf.namelist())
tgz = io.BytesIO(zf.read([n for n in zf.namelist() if n.endswith(".tar.gz")][0]))
tf = tarfile.open(fileobj=tgz, mode="r:gz")
members = {m.name.split("/")[-1]: m for m in tf.getmembers()}
print(list(members))
X_raw = pd.read_csv(tf.extractfile(members["data.csv"]), index_col=0)
y_raw = pd.read_csv(tf.extractfile(members["labels.csv"]), index_col=0)["Class"]
print("expression matrix:", X_raw.shape, "| labels:", y_raw.value_counts().to_dict())
X_raw.head(3).iloc[:, :6]""")

md("""## 2. Prepare the data
* log2(x+1) transform (RNA-seq counts are skewed; paper §3.2 requires a log transform).
* Drop genes with zero variance (non-active genes; paper §3.2 filters these).
* Stratified 80/20 split **before** any further fitting, so the 20 % test set stays completely unseen.
* Gene selection (top-N most variable), standardisation and optional PCA are all fitted on training data only.""")
code("""X_all = np.log2(X_raw.values + 1.0)
nonzero = X_all.var(axis=0) > 0
X_all = X_all[:, nonzero]
gene_names = X_raw.columns[nonzero]
le_classes = sorted(y_raw.unique())
y = y_raw.map({c: i for i, c in enumerate(le_classes)}).values
K = len(le_classes)
X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=SEED)
print("genes after zero-variance filter:", X_all.shape[1])
print("train:", X_tr_raw.shape, "| unseen test:", X_te_raw.shape)""")

md("""## 3. Hyper-parameter tuning with 5-fold cross-validation (training data only)
Each candidate pipeline (number of top-variance genes × number of PCA components) is fitted on 4 folds;
clusters are mapped to tumour types using **only those training folds' labels**, then scored on the held-out fold.
The test set is never touched here.""")
code("""class KMeansPipeline:
    \"\"\"top-variance genes -> StandardScaler -> (PCA) -> K-Means, cluster->class map learned on training labels.\"\"\"
    def __init__(self, n_genes, n_pcs, n_init=50):
        self.n_genes, self.n_pcs, self.n_init = n_genes, n_pcs, n_init
    def transform(self, X):
        Z = self.scaler.transform(X[:, self.top])
        return self.pca.transform(Z) if self.pca is not None else Z
    def fit(self, X, y):
        self.top = np.argsort(X.var(axis=0))[::-1][:self.n_genes]
        self.scaler = StandardScaler().fit(X[:, self.top])
        Z = self.scaler.transform(X[:, self.top])
        self.pca = PCA(self.n_pcs, random_state=SEED).fit(Z) if self.n_pcs else None
        Z = self.transform(X)
        self.km = KMeans(n_clusters=K, n_init=self.n_init, random_state=SEED).fit(Z)
        cm = np.zeros((K, K), dtype=int)
        for p, t in zip(self.km.labels_, y):
            cm[p, t] += 1
        r, c = linear_sum_assignment(-cm)              # Hungarian matching on TRAINING labels
        self.mapping = dict(zip(r, c))
        return self
    def clusters(self, X):
        return self.km.predict(self.transform(X))
    def predict(self, X):
        return np.array([self.mapping[p] for p in self.clusters(X)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for g in [500, 1000, 2000, 5000]:
    for pc in [None, 5, 10, 20, 50]:
        accs = [(KMeansPipeline(g, pc, n_init=20).fit(X_tr_raw[a], y_train[a]).predict(X_tr_raw[b]) == y_train[b]).mean()
                for a, b in skf.split(X_tr_raw, y_train)]
        rows.append({"n_genes": g, "n_pcs": pc or "none", "cv_acc_mean": np.mean(accs), "cv_acc_std": np.std(accs)})
        print(rows[-1])
cv = pd.DataFrame(rows).sort_values(["cv_acc_mean", "cv_acc_std"], ascending=[False, True]).reset_index(drop=True)
best = cv.iloc[0]
BEST_GENES = int(best.n_genes); BEST_PCS = None if best.n_pcs == "none" else int(best.n_pcs)
print("best:", BEST_GENES, "genes,", BEST_PCS, "PCs, CV accuracy", round(best.cv_acc_mean, 4))
cv.head(10)""")

md("## 4. Train the final K-Means (M11) on the full training set")
code("""pipe = KMeansPipeline(BEST_GENES, BEST_PCS, n_init=50).fit(X_tr_raw, y_train)
model, scaler = pipe.km, pipe.scaler
genes_kept = gene_names[pipe.top]
X_train_s, X_test_s = pipe.transform(X_tr_raw), pipe.transform(X_te_raw)
print("final feature space:", X_train_s.shape[1], "dims | inertia =", round(model.inertia_, 1))

# elbow / silhouette in the chosen feature space (supports k = 5)
ks = range(2, 11); inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_train_s)
    inertia.append(km.inertia_); sil.append(silhouette_score(X_train_s, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set(title="Elbow (inertia)", xlabel="k")
ax[1].plot(list(ks), sil, "o-"); ax[1].set(title="Silhouette (train)", xlabel="k")
plt.tight_layout(); plt.savefig("k_selection.png", dpi=150); plt.show()""")

md("""## 5. Test on unseen data and evaluate
Accuracy uses the cluster→class mapping learned from the **training** set (no peeking at test labels).
ARI, NMI and silhouette are label-permutation-free clustering metrics; precision/recall/F1 are per tumour type.""")
code("""from sklearn.metrics import classification_report, accuracy_score
def evaluate(Xr, Xs, y_true):
    cl = pipe.clusters(Xr); pred = pipe.predict(Xr)
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "ARI": float(adjusted_rand_score(y_true, cl)),
            "NMI": float(normalized_mutual_info_score(y_true, cl)),
            "silhouette": float(silhouette_score(Xs, cl))}, pred

train_metrics, _ = evaluate(X_tr_raw, X_train_s, y_train)
test_metrics, test_mapped = evaluate(X_te_raw, X_test_s, y_test)
test_metrics["n_test"] = int(len(y_test)); test_metrics["n_correct"] = int((test_mapped == y_test).sum())
test_metrics.update({"best_n_genes": BEST_GENES, "best_n_pcs": BEST_PCS})
results = pd.DataFrame({"train (seen)": train_metrics, "test (unseen)": test_metrics}).T
print(classification_report(y_test, test_mapped, target_names=le_classes, digits=4))
results""")

code("""test_cm = confusion_matrix(y_test, test_mapped)
cm_df = pd.DataFrame(test_cm, index=[f"true {c}" for c in le_classes],
                     columns=[f"pred {c}" for c in le_classes])
print("confusion matrix (unseen test set):")
cm_df""")

md("""## 5b. Robustness on a larger unseen set
The final 161-sample test set is small, so the chosen pipeline (hyper-parameters fixed from step 3) is retrained on **only 50 %
or only 20 %** of all 801 samples and evaluated on the remaining **401 / 641 unseen samples**, over 10 random stratified splits each.""")
code("""large_rows = []
for test_frac in [0.5, 0.8]:
    for rep in range(10):
        a_X, b_X, a_y, b_y = train_test_split(X_all, y, test_size=test_frac, stratify=y, random_state=100 + rep)
        pr = KMeansPipeline(BEST_GENES, BEST_PCS, n_init=20).fit(a_X, a_y).predict(b_X)
        large_rows.append({"train_samples": len(a_y), "unseen_test_samples": len(b_y), "repeat": rep,
                           "accuracy": accuracy_score(b_y, pr), "ARI": adjusted_rand_score(b_y, pr)})
large = pd.DataFrame(large_rows)
large_summary = large.groupby(["train_samples", "unseen_test_samples"])[["accuracy", "ARI"]].agg(["mean", "std", "min"])
large_summary""")

md("## 6. Visualise")
code("""pca = PCA(n_components=2, random_state=SEED).fit(X_train_s)
Z = pca.transform(X_test_s)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(Z[:, 0], Z[:, 1], c=test_mapped, cmap="tab10", s=14)
ax[0].set(title="Unseen test: K-Means clusters (PCA)", xlabel="PC1", ylabel="PC2")
ax[1].scatter(Z[:, 0], Z[:, 1], c=y_test, cmap="tab10", s=14)
ax[1].set(title="Unseen test: true tumour types (PCA)", xlabel="PC1", ylabel="PC2")
plt.tight_layout(); plt.savefig("clusters_pca.png", dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(test_cm, cmap="Blues")
ax.set_xticks(range(K), le_classes); ax.set_yticks(range(K), le_classes)
for (i, j), v in np.ndenumerate(test_cm):
    ax.text(j, i, v, ha="center", va="center")
ax.set(title="Confusion matrix, unseen test set", xlabel="predicted (mapped cluster)", ylabel="true class")
plt.tight_layout(); plt.savefig("confusion_test.png", dpi=150); plt.show()""")

md("## 7. Save model, metrics and artefacts")
code("""os.makedirs("model", exist_ok=True); os.makedirs("results", exist_ok=True)
joblib.dump({"kmeans": model, "scaler": scaler, "pca": pipe.pca, "genes": list(genes_kept),
             "cluster_to_class": {int(k): le_classes[v] for k, v in pipe.mapping.items()},
             "classes": le_classes}, "model/kmeans_pancan.joblib")
cv.to_csv("results/cv_tuning.csv", index=False)
large.to_csv("results/large_unseen_tests.csv", index=False)
results.to_csv("results/metrics.csv")
json.dump({"test": test_metrics, "train": train_metrics, "k": K, "seed": SEED},
          open("results/metrics.json", "w"), indent=2)
cm_df.to_csv("results/confusion_test.csv")
print(open("results/metrics.json").read())
from google.colab import files
for f in ["model/kmeans_pancan.joblib", "results/metrics.json", "results/metrics.csv",
          "results/confusion_test.csv", "results/cv_tuning.csv", "results/large_unseen_tests.csv", "k_selection.png", "clusters_pca.png", "confusion_test.png"]:
    files.download(f)""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("kmeans_gene_expression.ipynb")
nbf.write(nb, out)
print("wrote", out)
