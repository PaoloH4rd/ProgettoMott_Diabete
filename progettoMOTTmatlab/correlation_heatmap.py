import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caricamento dataset
file_path = '../progettoMOTTmatlab/Dataset_Diabete_Completo_Pulito.csv'
df = pd.read_csv(file_path)

# 🔴 Rimuovi SEQN
df = df.drop(columns=['SEQN'])

# Seleziona solo colonne numeriche
df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Matrice di correlazione
corr_matrix = df_numeric.corr(method='pearson')

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5
)

plt.title("Correlation Heatmap - Dataset Diabete (senza SEQN)", fontsize=14)
plt.tight_layout()

# Salvataggio su file (consigliato)
plt.savefig("correlation_heatmap_diabete_no_seqn.png", dpi=300)
plt.close()

print("Heatmap salvata come 'correlation_heatmap_diabete_no_seqn.png'")