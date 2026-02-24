# Analisi Statistica e Predittiva dei Dati sul Diabete
**Progetto 8 – Modelli e Metodi di Ottimizzazione Statistica**

---

## 1. Introduzione e Obiettivi del Progetto

Il diabete mellito rappresenta una delle sfide sanitarie globali più critiche del XXI secolo. La capacità di prevedere l'insorgenza della malattia o monitorarne la progressione attraverso variabili cliniche misurabili è fondamentale per la prevenzione e il trattamento precoce.

Il presente progetto si pone l'obiettivo di sviluppare un modello statistico predittivo in grado di stimare i livelli di **Glicoemoglobina (HbA1c)**, un biomarcatore cruciale per la diagnosi del diabete, partendo da un set eterogeneo di variabili fisiologiche e comportamentali.

Utilizzando i dati provenienti dal **National Health and Nutrition Examination Survey (NHANES)**, il lavoro è stato strutturato per affrontare specifiche sfide di eterogeneità dei dati e validazione statistica. L'approccio integrato Python-MATLAB ha permesso di trasformare dati grezzi complessi in un modello di regressione lineare robusto e validato.

---

## 2. Metodologia di Data Engineering (Python)

La fase preliminare di elaborazione dati è stata eseguita mediante lo script `process_nhanes.py`. L'approccio adottato ha seguito una pipeline rigorosa per trasformare dati grezzi disaggregati in un dataset analitico coerente, risolvendo criticità strutturali significative.

### 2.1 Acquisizione e Integrazione delle Fonti
I dati originali erano frammentati in molteplici file CSV, ciascuno rappresentante un dominio clinico specifico. L'integrazione è avvenuta utilizzando il codice identificativo univoco del paziente (`SEQN`) come chiave primaria di join. Le variabili selezionate includono:

*   **Variabile Target (Y):**
    *   `LBXGH` (Glicoemoglobina): La misura target per la valutazione del controllo glicemico a lungo termine.
*   **Covariate (X):**
    *   `RIDAGEYR` (Età) e `RIAGENDR` (Genere): Fattori demografici di base.
    *   `BMXBMI` (Body Mass Index): Indicatore chiave per l'obesità.
    *   `LBXGLU` (Glucosio Plasmatico a Digiuno) e `LBXIN` (Insulina): Indicatori metabolici diretti.
    *   `LBXTC` (Colesterolo Totale): Indicatore di salute cardiovascolare.

### 2.2 La Sfida dell'Attività Fisica: Standardizzazione e Normalizzazione
Una delle problematiche principali riscontrate riguardava la variabile dell'attività fisica, registrata nei database NHANES in modo eterogeneo su due colonne: `PAD790Q` (quantità numerica) e `PAD790U` (unità temporale: giorni, settimane, mesi, anni). Questa frammentazione rendeva impossibile il confronto diretto tra pazienti (es. confrontare "2 volte l'anno" con "3 volte la settimana").

Per risolvere questo problema, è stato implementato un algoritmo di normalizzazione (`process_physical_activity`) che unifica queste informazioni in una singola metrica standardizzata: **Frequenza Settimanale (LTPA Weekly Freq)**.
La logica di conversione applicata è stata la seguente:
*   **Giornaliera (D):** Moltiplicazione per 7.
*   **Settimanale (W):** Valore invariato.
*   **Mensile (M):** Divisione per 4.33 (media settimane/mese).
*   **Annuale (Y):** Divisione per 52.

### 2.3 Pulizia dei Dati (Data Cleaning)
Per garantire la convergenza del modello statistico, sono state applicate rigide procedure di pulizia:
1.  **Rimozione Valori Anomali:** Sono stati filtrati i codici di errore NHANES (es. 7777 "Rifiutato", 9999 "Non so") e artefatti numerici (es. valori infinitesimali quali `5.39e-79`).
2.  **Gestione dei Dati Mancanti (NaN):** La presenza di valori mancanti impediva il calcolo algebrico delle matrici di regressione. È stata adottata una strategia di "complete case analysis", rimuovendo i record incompleti.
3.  **Filtraggio di Coerenza:** Sono stati mantenuti solo i soggetti con frequenza di attività fisica rappresentata da numeri interi positivi (`> 0`), eliminando valori frazionari o nulli per ridurre il rumore.

---

## 3. Analisi Statistica e Modellazione (MATLAB)

La fase analitica (`ProgettoMOTT.m`) si è concentrata sulla quantificazione delle relazioni tra le variabili e la costruzione del modello predittivo.

### 3.1 Analisi delle Correlazioni
Prima della modellazione, è stata effettuata un'analisi esplorativa calcolando i coefficienti di correlazione di Pearson ($r$) tra ogni predittore e la Glicoemoglobina.
*   **Correlazione Positiva Attesa ($r > 0$):** Variabili come BMI, Glucosio a digiuno e Età mostrano una crescita concordante con il rischio diabetico.
*   **Correlazione Negativa Attesa ($r < 0$):** L'attività fisica (LTPA) standardizzata ha mostrato, come ipotizzato, un effetto protettivo inverso rispetto alla malattia.

### 3.2 Regressione Lineare Multipla (MLR) e la "Trappola del P-Value"
Il modello di regressione lineare multipla è stato definito come:
$$ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon $$

Nelle fasi iniziali, il modello è stato addestrato sull'intero dataset senza suddivisioni. I risultati mostravano un **p-value pari a 0** (o estremamente prossimo allo zero) per il modello globale (F-test). Sebbene questo indicasse una forte significatività statistica, il risultato era fuorviante ai fini predittivi: un p-value nullo su tutto il dataset confermava solo le correlazioni nel campione osservato, ma non offriva garanzie sulla capacità di diagnosticare nuovi pazienti (rischio di *overfitting*).

Questa constatazione ha reso necessario l'abbandono della valutazione puramente "in-sample" a favore di una strategia di validazione più rigorosa.

---

## 4. Approccio di Validazione del Modello

Per superare i limiti dell'analisi preliminare e misurare la reale capacità di generalizzazione del modello, è stato implementato un protocollo di validazione "Out-of-Sample".

### 4.1 Strategia di Suddivisione (Holdout 80/20)
Il dataset pulito è stato partizionato casualmente in due sottoinsiemi disgiunti:
*   **Training Set (80%):** Utilizzato per la stima dei coefficienti $\beta$, garantendo una base dati sufficientemente ampia per ridurre la varianza delle stime.
*   **Test Set (20%):** Mantenuto "cieco" durante l'addestramento. Questo sottoinsieme simula l'arrivo di nuovi pazienti clinici e serve esclusivamente per la verifica finale.

### 4.2 Valutazione Visiva delle Predizioni (Scatter Plot)
Una difficoltà operativa significativa riguardava l'interpretazione intuitiva dell'errore numerico. Per valutare qualitativamente le performance, è stato generato uno **Scatter Plot (Valori Reali vs. Valori Predetti)** sui dati di test.

L'analisi grafica ha mostrato che i punti si dispongono lungo la **bisettrice del primo e terzo quadrante** (la retta $y=x$). Questa configurazione geometrica è la conferma visiva che $y_{predetto} \approx y_{reale}$. La densità dei punti attorno a questa retta, senza pattern sistematici di deviazione, ha permesso di validare immediatamente la bontà del modello, dimostrando che le predizioni seguono fedelmente il trend clinico reale e non sono frutto del caso.

### 4.3 Metriche Quantitative: RMSE e R-quadro Out-of-Sample
Oltre all'analisi visiva, l'accuratezza è stata quantificata tramite metriche oggettive calcolate sul Test Set:

1.  **RMSE (Root Mean Squared Error):** Misura la deviazione standard dei residui di predizione, indicando l'errore medio in unità della variabile target.
2.  **R-quadro ($R^2$) Out-of-Sample:**
    A differenza dell'$R^2$ classico, questo indicatore è calcolato confrontando l'errore del modello con la varianza totale del test set ($1 - SSE/SST$). Un valore positivo conferma che il modello riesce a spiegare una porzione significativa della varianza della glicoemoglobina anche su pazienti mai visti prima, validando l'efficacia clinica delle variabili selezionate e la robustezza dell'approccio statistico adottato.
