import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis():
    # 1. Caricamento dei dati
    print("Caricamento dei file...")
    try:
        demo = pd.read_csv('DEMO_L.csv')[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
        bmi = pd.read_csv('Body_Measures(BMI).csv')[['SEQN', 'BMXBMI', 'BMXWAIST']]
        glucose = pd.read_csv('Glucosio_plasmatico.csv')[['SEQN', 'LBXGLU']]
        insulin = pd.read_csv('insulina.csv')[['SEQN', 'LBXIN']]
        cholesterol = pd.read_csv('Colesterolo_totale.csv')[['SEQN', 'LBXTC']]
        hba1c = pd.read_csv('Glicoemoglobina.csv')[['SEQN', 'LBXGH']]
    except Exception as e:
        print(f"Errore nel caricamento dei file: {e}")
        return

    # 2. Merge dei dataset
    print("Unione dei dataset...")
    df = hba1c.merge(demo, on='SEQN', how='inner')
    df = df.merge(bmi, on='SEQN', how='inner')
    df = df.merge(glucose, on='SEQN', how='inner')
    df = df.merge(insulin, on='SEQN', how='inner')
    df = df.merge(cholesterol, on='SEQN', how='inner')

    # 3. Pulizia dei dati
    # Rimuoviamo i valori nulli per la regressione
    initial_len = len(df)
    df = df.dropna()
    print(f"Dati puliti: rimossi {initial_len - len(df)} record con valori mancanti. Record rimanenti: {len(df)}")

    # Salvataggio del dataset finale unito
    df.to_csv('dataset_merged.csv', index=False)
    print("Dataset unito salvato come 'dataset_merged.csv'")

    # 4. Regressione Lineare Multipla
    # Variabile Target (Y)
    y = df['LBXGH']
    
    # Variabili Predittive (X)
    X = df[['RIDAGEYR', 'BMXBMI', 'BMXWAIST', 'LBXGLU', 'LBXIN', 'LBXTC']]
    
    # Aggiungiamo la costante (intercetta beta_0)
    X = sm.add_constant(X)

    # Fit del modello
    model = sm.OLS(y, X).fit()

    # 5. Output dei risultati
    print("\n" + "="*50)
    print("RISULTATI DELLA REGRESSIONE LINEARE MULTIPLA")
    print("="*50)
    print(model.summary())
    
    # 6. Salvataggio dei risultati in un file di testo per consultazione
    with open('risultati_regressione.txt', 'w') as f:
        f.write(model.summary().as_text())
    print("\nI risultati dettagliati sono stati salvati in 'risultati_regressione.txt'")

    # 7. Grafico di correlazione (Heatmap)
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[['LBXGH', 'RIDAGEYR', 'BMXBMI', 'BMXWAIST', 'LBXGLU', 'LBXIN', 'LBXTC']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matrice di Correlazione tra Variabili')
        plt.savefig('correlazione.png')
        print("Grafico di correlazione salvato come 'correlazione.png'")
    except Exception as e:
        print(f"Errore nella generazione del grafico: {e}")

if __name__ == "__main__":
    run_analysis()