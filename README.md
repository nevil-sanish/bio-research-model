# bio-research-model

ISC 211 Introduction to Bioinformatics, individual assignment by Nevil Mathew Sanish (2025BCD0011).

**Topic 11:** Gene Expression Clustering and Analysis (Georgieva, ERIS 2022). **Algorithm M11:** K-Means.

K-Means groups 801 TCGA tumour RNA-seq samples (UCI dataset 401) into 5 clusters without using labels. The labels are used only to evaluate the clusters.

| Version | Unseen-test accuracy |
|---|---|
| v1 baseline (2,000 genes) | 98.76% |
| v2 CV-tuned (500 genes) | 99.38% (160/161) |
| v3 large unseen test | 99.77% on 641 unseen / 99.80% on 401 unseen |

## Layout
- `dataset/`: the UCI cancer RNA-seq archive, plus `paper_bottomly_mouse_rnaseq/` (the paper's own mouse dataset, from ReCount).
- `model/build_notebook.py`: generates `kmeans_gene_expression.ipynb`. Run that notebook in Google Colab.
- `model/version*/`: one folder per version, each with the notebook (with outputs), the saved model, results and figures.
- `report/`: the report (`.docx`/`.pdf`) and `build_report.py`.
