import os
import textwrap
import pandas as pd
import numpy as np
np.seterr(all="ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from fpdf import FPDF
from kmedoids_py import kmedoids
import warnings
warnings.filterwarnings("ignore")


# ---------- File Paths Configuration ----------
RFM_CSV   = "data/processed/rfm.csv"
TRANS_CSV = "data/processed/clean_transactions.csv"
FIG_DIR   = "reports/figures"
OUT_CSV   = "data/processed/clustered_rfm.csv"
PDF_PATH  = "reports/Project_Report.pdf"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Helper Functions ----------
def pca_scatter(X, labels, title, fname):
    """
    Generate and save a 3D scatter plot of PCA-transformed data
    
    Parameters:
    X (numpy.ndarray): Input data
    labels (numpy.ndarray): Cluster labels
    title (str): Plot title
    fname (str): Output file path
    """
    pts = PCA(3, random_state=1).fit_transform(X)
    fig = plt.figure(figsize=(6,5)); ax = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=labels, cmap="tab10", s=20, alpha=.8)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    cb = fig.colorbar(sc, ax=ax, shrink=.6); cb.set_label("cluster")
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()

# ---------- 1. Data Loading ----------
rfm = pd.read_csv(RFM_CSV, index_col=0)
X   = rfm[["Recency","Frequency","Monetary"]].values

# ---------- 2. Clustering Analysis ----------
# Hierarchical single-linkage clustering (Euclidean distance)
Z         = linkage(X, method="single", metric="euclidean")
hier_lbl  = fcluster(Z, 4, "maxclust")
plt.figure(figsize=(9,3.5))
dendrogram(Z, truncate_mode="lastp", p=25, show_contracted=True,
            leaf_rotation=90, leaf_font_size=8)
plt.title("Truncated dendrogram (single linkage)"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/dendrogram.png", dpi=120); plt.close()

# K-Medoids clustering (Manhattan distance)
kmed_lbl  = kmedoids(X, k=4)

# Calculate silhouette scores for both clustering algorithms
h_sil  = silhouette_score(X, hier_lbl, metric="euclidean")
km_sil = silhouette_score(X, kmed_lbl, metric="manhattan")
plt.figure(figsize=(4,3))
plt.bar(["Hierarchical", "K-Medoids"], [h_sil, km_sil], color=["#4c72b0","#dd8452"])
plt.title("Silhouette score comparison"); plt.ylabel("Score (0-1)")
for i,v in enumerate([h_sil, km_sil]): plt.text(i, v+.01, f"{v:.2f}", ha="center")
plt.ylim(0, max(h_sil, km_sil)+.1); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/silhouette.png", dpi=120); plt.close()

# Generate PCA visualizations for both clustering methods
pca_scatter(X, hier_lbl, "Hierarchical Clustering (k=4)", f"{FIG_DIR}/pca_hier.png")
pca_scatter(X, kmed_lbl, "K-Medoids Clustering (k=4)", f"{FIG_DIR}/pca_km.png")

# Generate elbow curve using KMeans as baseline
sse = [KMeans(k, n_init="auto", random_state=1).fit(X).inertia_ for k in range(1,11)]
plt.figure(figsize=(4,3)); plt.plot(range(1,11), sse, marker="o")
plt.title("Elbow curve (KMeans)"); plt.xlabel("Number of clusters (k)"); plt.ylabel("Sum of Squared Errors")
plt.xticks(range(1,11))
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/elbow.png", dpi=120); plt.close()

# ---------- 3. Save Cluster Results ----------
rfm["hier_cluster"]     = hier_lbl
rfm["kmedoids_cluster"] = kmed_lbl
rfm.to_csv(OUT_CSV)

# ---------- 4. Association Rule Mining (FP-growth) ----------
tx = pd.read_csv(TRANS_CSV)
basket = (tx.groupby(["Invoice","Description"])["Quantity"].sum()
            .unstack(fill_value=0).gt(0))
freq = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1)
rules.to_csv("data/processed/association_rules.csv", index=False)

# ---------- 5. Cluster Visualization: Relative Importance Heatmap ----------

# Re-load the latest clustered data (ensure clusters are there)
rfm = pd.read_csv(OUT_CSV)

# Check if 'kmedoids_cluster' column exists and contains values
if "kmedoids_cluster" not in rfm.columns or rfm["kmedoids_cluster"].nunique() < 2:
    raise ValueError("Missing or invalid 'kmedoids_cluster' column for heatmap visualization.")

# Calculate average RFM values per cluster and overall population
cluster_avg = rfm.groupby("kmedoids_cluster")[["Recency","Frequency","Monetary"]].mean()
pop_avg     = rfm[["Recency","Frequency","Monetary"]].mean()

# Prevent division by zero and NaNs
rel_imp = (cluster_avg / pop_avg.replace(0, np.nan)) - 1
rel_imp = rel_imp.replace([np.inf, -np.inf], np.nan).fillna(0)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(rel_imp.round(2), annot=True, cmap="RdYlGn", center=0,
            cbar_kws=dict(label="Relative importance"),
            annot_kws={"size": 12}, fmt=".2f")
plt.title("Attribute Importance per K-Medoids Cluster", fontsize=14)
plt.ylabel("Cluster", fontsize=12)
plt.xlabel("Attribute", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/relimp.png", dpi=150)
plt.close()

# ---------- 6. Generate PDF Report ----------
# Initialize PDF with larger page size and adequate margins
pdf = FPDF(orientation='P', unit='mm', format='A4')
pdf.set_auto_page_break(True, margin=15)
pdf.set_margins(left=20, top=15, right=20)
pdf.add_page()

# Add title with appropriate font size
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 10, "Project Report - Online Retail II Clustering Analysis", 0, 1, 'C')

# Set smaller font size for content
pdf.set_font("Helvetica", "", 10)
pdf.ln(5)

# Use individual cells instead of multi_cell for safer text rendering
pdf.cell(0, 6, f"Dataset size    : {len(tx):,} rows  /  {tx['Customer ID'].nunique():,} customers", 0, 1)
pdf.cell(0, 6, f"Clusters        : Hierarchical vs K-Medoids, both k=4", 0, 1)
pdf.cell(0, 6, f"Silhouette      : Hierarchical {h_sil:.2f}   /   K-Medoids {km_sil:.2f}", 0, 1)
pdf.cell(0, 6, f"FP-Growth       : {len(freq):,} item-sets  ->  {len(rules):,} rules  (>=2% support)", 0, 1)

def add_fig(path, caption):
    """Add a figure to the PDF report with a caption"""
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, caption, 0, 1, 'C')
    # Position image with safe margins
    pdf.image(path, x=20, y=30, w=170)

# Add figures to the report
add_fig(f"{FIG_DIR}/dendrogram.png",   "Figure 1: Truncated Dendrogram (Single Linkage)")
add_fig(f"{FIG_DIR}/pca_hier.png",     "Figure 2: Hierarchical Clusters Visualization (PCA)")
add_fig(f"{FIG_DIR}/pca_km.png",       "Figure 3: K-Medoids Clusters Visualization (PCA)")
add_fig(f"{FIG_DIR}/silhouette.png",   "Figure 4: Silhouette Scores Comparison")
add_fig(f"{FIG_DIR}/elbow.png",        "Figure 5: Elbow Curve (KMeans Baseline)")
add_fig(f"{FIG_DIR}/relimp.png",       "Figure 6: Relative Importance Heat-Map")

# Save PDF report
pdf.output(PDF_PATH)

print(f"[load] Cluster results saved to: {OUT_CSV}")
print(f"[load] Figures saved to: {FIG_DIR}")
print(f"[load] PDF report generated: {PDF_PATH}")
