import pandas as pd
import numpy as np
import os

def process_physical_activity(file_path):
    """
    Processa le variabili NHANES PAD790Q (Quantità) e PAD790U (Unità)
    per creare una frequenza settimanale standardizzata.
    """
    
    # Caricamento dati
    df = pd.read_csv(file_path)
    
    # 1. Gestione dei valori mancanti/speciali (Cleaning)
    # NHANES usa 7777 (Refused) e 9999 (Don't Know)
    # Inoltre, nel CSV fornito appare un valore molto piccolo (es. 5.397605e-79) che trattiamo come NaN
    epsilon = 5.397605346934028e-79
    df['PAD790Q'] = df['PAD790Q'].replace({7777: np.nan, 9999: np.nan, epsilon: np.nan})
    
    # 2. Creazione della nuova variabile (Feature Engineering)
    # Inizializziamo a NaN
    df['LTPA_Weekly_Freq'] = np.nan
    
    # Normalizziamo la colonna unità in stringa per gestire i formati b'W', b'D', ecc.
    df['PAD790U'] = df['PAD790U'].astype(str)
    
    # Logica di conversione
    # Mappatura basata sui valori osservati nel CSV:
    # b'D' -> Giorno (Day)
    # b'W' -> Settimana (Week)
    # b'M' -> Mese (Month)
    # b'Y' -> Anno (Year)
    
    # Caso 1: Giornaliero -> Moltiplica per 7
    mask_day = df['PAD790U'].str.contains('D')
    df.loc[mask_day, 'LTPA_Weekly_Freq'] = df.loc[mask_day, 'PAD790Q'] * 7
    
    # Caso 2: Settimanale -> Rimane uguale
    mask_week = df['PAD790U'].str.contains('W')
    df.loc[mask_week, 'LTPA_Weekly_Freq'] = df.loc[mask_week, 'PAD790Q']
    
    # Caso 3: Mensile -> Dividi per ~4.33 (settimane in un mese)
    mask_month = df['PAD790U'].str.contains('M')
    df.loc[mask_month, 'LTPA_Weekly_Freq'] = df.loc[mask_month, 'PAD790Q'] / 4.33
    
    # Caso 4: Annuale -> Dividi per 52
    mask_year = df['PAD790U'].str.contains('Y')
    df.loc[mask_year, 'LTPA_Weekly_Freq'] = df.loc[mask_year, 'PAD790Q'] / 52
    
    return df

def merge_datasets():
    """
    Unisce i vari dataset NHANES in un unico DataFrame finale.
    """
    print("Inizio fusione dataset...")
    
    # 1. Caricamento Dataset
    # Glicoemoglobina (Target)
    df_gh = pd.read_csv('Glicoemoglobina.csv')[['SEQN', 'LBXGH']]
    df_gh.rename(columns={'LBXGH': 'Glicoemoglobina'}, inplace=True)
    
    # Demografici (Età, Genere)
    # RIAGENDR: 1 = Maschio, 2 = Femmina
    # RIDAGEYR: Età in anni
    df_demo = pd.read_csv('DEMO_L.csv')[['SEQN', 'RIAGENDR', 'RIDAGEYR']]
    df_demo.rename(columns={'RIAGENDR': 'Genere', 'RIDAGEYR': 'Eta'}, inplace=True)
    
    # Body Measures (BMI)
    # BMXBMI: Body Mass Index
    df_bmi = pd.read_csv('Body_Measures(BMI).csv')[['SEQN', 'BMXBMI']]
    df_bmi.rename(columns={'BMXBMI': 'BMI'}, inplace=True)
    
    # Glucosio Plasmatico (Glicemia a digiuno)
    # LBXGLU: Fasting Glucose (mg/dL)
    df_glu = pd.read_csv('Glucosio_plasmatico.csv')[['SEQN', 'LBXGLU']]
    df_glu.rename(columns={'LBXGLU': 'Glicemia_Digiuno'}, inplace=True)
    
    # Insulina
    # LBXIN: Insulin (uU/mL)
    df_ins = pd.read_csv('insulina.csv')[['SEQN', 'LBXIN']]
    df_ins.rename(columns={'LBXIN': 'Insulina'}, inplace=True)
    
    # Colesterolo Totale
    # LBXTC: Total Cholesterol (mg/dL)
    df_chol = pd.read_csv('Colesterolo_totale.csv')[['SEQN', 'LBXTC']]
    df_chol.rename(columns={'LBXTC': 'Colesterolo_Totale'}, inplace=True)
    
    # Attività Fisica (Processata)
    if os.path.exists('Attività_fisica_processed.csv'):
        df_pa = pd.read_csv('Attività_fisica_processed.csv')
    else:
        # Se non esiste, lo rigeneriamo al volo
        print("Attività_fisica_processed.csv non trovato, rigenerazione in corso...")
        df_pa_raw = process_physical_activity('Attività_fisica.csv')
        df_pa = df_pa_raw[['SEQN', 'LTPA_Weekly_Freq']]
    
    # 2. Fusione (Merge)
    # Usiamo 'SEQN' come chiave.
    # Iniziamo dal target (Glicoemoglobina) perché ci interessano i pazienti che hanno questo valore.
    # Usiamo 'inner' o 'left' join a seconda se vogliamo tenere solo chi ha tutti i dati o no.
    # Per ora usiamo merge sequenziali.
    
    df_final = df_gh.merge(df_demo, on='SEQN', how='inner')
    df_final = df_final.merge(df_bmi, on='SEQN', how='left')
    df_final = df_final.merge(df_glu, on='SEQN', how='left')
    df_final = df_final.merge(df_ins, on='SEQN', how='left')
    df_final = df_final.merge(df_chol, on='SEQN', how='left')
    df_final = df_final.merge(df_pa, on='SEQN', how='left')
    
    return df_final

if __name__ == "__main__":
    # Step 1: Processa Attività Fisica se necessario
    file_pa_raw = 'Attività_fisica.csv'
    if os.path.exists(file_pa_raw):
        print(f"Elaborazione del file attività fisica: {file_pa_raw}...")
        df_processed = process_physical_activity(file_pa_raw)
        df_pa_final = df_processed[['SEQN', 'LTPA_Weekly_Freq']]
        df_pa_final.to_csv('Attività_fisica_processed.csv', index=False)
        print("Attività fisica processata e salvata.")
    
    # Step 2: Unisci tutto
    try:
        df_dataset = merge_datasets()
        
        print("\nDataset Finale Creato (Prima della pulizia NaN):")
        print(f"Dimensioni: {df_dataset.shape}")
        
        # Step 3: Rimuovi righe con NaN
        df_dataset_clean = df_dataset.dropna()
        
        # Step 4: Filtra valori LTPA
        # Rimuoviamo:
        # - Valori 0
        # - Valori frazionari (sia < 1 che > 1)
        # Manteniamo solo interi positivi.
        
        # Filtro per interi (resto della divisione per 1 deve essere 0)
        # Nota: usiamo una tolleranza minima per float arithmetic se necessario, ma qui % 1 == 0 è standard per check interi
        df_dataset_clean = df_dataset_clean[df_dataset_clean['LTPA_Weekly_Freq'] % 1 == 0]
        
        # Filtro per > 0
        df_dataset_clean = df_dataset_clean[df_dataset_clean['LTPA_Weekly_Freq'] > 0]
        
        print("\nDataset Finale Pulito (Senza NaN, senza 0, solo interi):")
        print(df_dataset_clean.head())
        print(f"Dimensioni: {df_dataset_clean.shape}")

        # Step 4.5: Salvataggio del dataset completo e pulito
        output_dir = '../progettoMOTTmatlab'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        complete_output_path = os.path.join(output_dir, 'Dataset_Diabete_Completo.csv')
        df_dataset_clean.to_csv(complete_output_path, index=False)
        print(f"\nDataset completo e pulito salvato in: {os.path.abspath(complete_output_path)}")

        # Step 5: Holdout (Train/Test Split)
        print("\nCreazione Holdout: 80% Train, 20% Test...")

        # 1. Mischia il dataset
        df_shuffled = df_dataset_clean.sample(frac=1, random_state=42) # random_state per riproducibilità

        # 2. Dividi in train e test
        train_size = int(0.8 * len(df_shuffled))
        train_set = df_shuffled[:train_size]
        test_set = df_shuffled[train_size:]

        print(f"Dimensioni Train Set: {train_set.shape}")
        print(f"Dimensioni Test Set: {test_set.shape}")

        # 6. Salvataggio dei nuovi dataset
        # Definiamo la cartella di output per i dati di train/test
        output_dir = '../progettoMOTTmatlab'
        
        # Creiamo la cartella se non esiste
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Cartella di output creata: {os.path.abspath(output_dir)}")

        # Definiamo i percorsi completi per i file
        train_output_path = os.path.join(output_dir, 'train_dataset.csv')
        test_output_path = os.path.join(output_dir, 'test_dataset.csv')

        # Salviamo i dataset
        train_set.to_csv(train_output_path, index=False)
        print(f"Train set salvato correttamente in: {os.path.abspath(train_output_path)}")

        test_set.to_csv(test_output_path, index=False)
        print(f"Test set salvato correttamente in: {os.path.abspath(test_output_path)}")
        
    except Exception as e:
        print(f"\nErrore durante la creazione del dataset: {e}")
