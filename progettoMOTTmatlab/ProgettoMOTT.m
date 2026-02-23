%% Svolgimento progetto 8 Modelli e Metodi di Ottimizzazione statistica di Paolo Ardolino, Giovanni Cafarelli, Nicola Inchingolo
data = readmatrix("Dataset_Diabete_Completo.csv");

% ================= DATASET COMPLETO =================
Y = data(:, 2);  % Assuming the second column corresponds to Glycohemoglobin
X1 = data(:, 3); % Assuming the third column corresponds to Gender
X2 = data(:, 4); % Assuming the fourth column corresponds to Age
X3 = data(:, 5); % Assuming the fifth column corresponds to Body Mass Index
X4 = data(:, 6); % Assuming the sixth column corresponds to Fasting Plasma Glucose
X5 = data(:, 7); % Assuming the seventh column corresponds to Insulin
X6 = data(:, 8); % Assuming the eighth column corresponds to Total Cholesterol
X7 = data(:, 9); % Assuming the nineth column corresponds to LTPA Weekly frequency


% regressione lineare multipla
tbl = table(X1, X2, X3, X4, X5, X6, X7, Y, 'VariableNames', {'Gender', 'Age', 'BMI', 'FastingGlucose', 'Insulin', 'Cholesterol', 'LTPA', 'Glycohemoglobin'});

lm = fitlm(tbl, 'Glycohemoglobin ~ Age + BMI + Gender + FastingGlucose + Insulin + Cholesterol + LTPA')
plot(lm)
% ---- VALUTAZIONE MODELLO ----
disp(" ")
disp("---- VALUTAZIONE MODELLO ----")
% ---- COEFFICIENTI DI CORRELAZIONE ----
% --- INTERPRETAZIONE DEL COEFFICIENTE DI CORRELAZIONE ---
% r > 0 : Correlazione POSITIVA (le variabili crescono insieme)
% r < 0 : Correlazione NEGATIVA (una cresce, l'altra decresce)
% r = 0 : ASSENZA di correlazione lineare
disp(" ")
disp("---- COEFFICIENTI DI CORRELAZIONE ----")
% coefficienti di correlazione:
corr_coeff1 = corr(X1,Y);
disp(" ")
disp("Coefficiente di correlazione con il genere")
disp(corr_coeff1)

corr_coeff2 = corr(X2,Y);
disp("Coefficiente di correlazione con l'età")
disp(corr_coeff2)

corr_coeff3 = corr(X3, Y);
disp("Coefficiente di correlazione con l'indice di massa corporea")
disp(corr_coeff3);

corr_coeff4 = corr(X4, Y);
disp("Coefficiente di correlazione con la glicemia a digiuno")
disp(corr_coeff4)

corr_coeff5 = corr(X5, Y);
disp("Coefficiente di correlazione con l'insulina")
disp(corr_coeff5)

corr_coeff6 = corr(X6, Y);
disp("Coefficiente di correlazione con il colesterolo totale")
disp(corr_coeff6);


corr_coeff7 = corr(X7, Y);
disp("Coefficiente di correlazione con la frequenza LTPA settimanale")
disp(corr_coeff7)

% ---- R^2 ----
disp(" ")
disp("---- R^2 ----")
disp(" ")
R2_adj = lm.Rsquared.Adjusted;
disp("R^2 aggiustato:")
disp(R2_adj)

if  R2_adj < 0.5
    disp("Il modello non spiega sufficientemente la variabilità dei dati.");
else
    disp("Il modello spiega adeguatamente la variabilità dei dati.");
end

% ---- F-TEST ----
disp(" ")
disp("---- F-TEST ----")
anovaResult = anova(lm, 'summary')

%studio l'Fstat
Fstat = lm.ModelFitVsNullModel.Fstat;
fcrit95 = finv(0.95, length(X1), anovaResult.DF(3))
disp(" ")
disp("la F-test presenta il seguente valore: ")
disp(Fstat)
disp("la F a livello di significatività presenta il seguente valore: ")
disp(fcrit95)


if Fstat > fcrit95
   disp("Il modello potrebbe essere sigbnificativo per il nostro dataset");
else
   disp("Il modello non potrebbe essere significativo per il nostro dataset");
end

% ---- p-value ----
disp(" ")
disp("---- p-value ----")
%studio il p-value
disp(" ")
pValue = anovaResult.pValue(2);
disp("il pValue presenta il seguente valore: ");
disp(pValue);


%il p-value del modello è molto basso rispetto al critico (0,05) il che
%indica che il modello è significativo per i dati

if pValue < 0.005
 disp("Il modello potrebbe essere sigbnificativo per il nostro dataset");
else
   disp("Il modello non potrebbe essere significativo per il nostro dataset");
end

% ---- NORMALITÀ RESIDUI ----
disp(" ")
disp("---- NORMALITÀ RESIDUI ----")
%eseguo test di normalità -> utilizzo Jarque-Bera
residui = lm.Residuals.Raw;
[h, p] = jbtest(residui)

if h == 0
    disp("Il test di Jarque-Bera NON rifiuta l'ipotesi di normalità dei residui.");
    disp("Le inferenze statistiche sul modello possono considerarsi affidabili.");
else
    disp("Il test di Jarque-Bera rifiuta l'ipotesi di normalità dei residui.");
    disp("Le inferenze statistiche sul modello potrebbero non essere affidabili.");
end

%Essendo il p-value di un valore molto piccolo, e l'F-test di unvalore
%molto grande, la nostra analisi non è sufficiente a verificare la bontà
%vera e propria del modello, trattando un dataset molto grande.

%Si è deciso, infatti, di proseguire l'analisi andando a suddividere il
%dataset in Train e test, effetuando le relative predizioni, realizzando
%uno scatter plot per visualizzare il plot, e guardando l'RMSE e
%l'R2-out-of-sample



%% Svolgimento progetto 8 Modelli e Metodi di Ottimizzazione statistica di Paolo Ardolino, Giovanni Cafarelli, Nicola Inchingolo - Analisi approfondita
% ================= TRAIN SET =================
train_data = readmatrix("train_dataset.csv");

Y_tr = train_data(:, 2);  % Assuming the second column corresponds to Glycohemoglobin
X1_tr = train_data(:, 3); % Assuming the third column corresponds to Gender
X2_tr = train_data(:, 4); % Assuming the fourth column corresponds to Age
X3_tr = train_data(:, 5); % Assuming the fifth column corresponds to Body Mass Index
X4_tr = train_data(:, 6); % Assuming the sixth column corresponds to Fasting Plasma Glucose
X5_tr = train_data(:, 7); % Assuming the seventh column corresponds to Insulin
X6_tr = train_data(:, 8); % Assuming the eighth column corresponds to Total Cholesterol
X7_tr = train_data(:, 9); % Assuming the nineth column corresponds to LTPA Weekly frequency


% regressione lineare multipla
tbl = table(X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, Y_tr, 'VariableNames', {'Gender', 'Age', 'BMI', 'FastingGlucose', 'Insulin', 'Cholesterol', 'LTPA', 'Glycohemoglobin'});

lm = fitlm(tbl, 'Glycohemoglobin ~ Age + BMI + Gender + FastingGlucose + Insulin + Cholesterol + LTPA')

plot(lm)
% ---- VALUTAZIONE TRAIN ----
disp(" ")
disp("---- VALUTAZIONE MODELLO TRAIN ----")

% ---- COEFFICIENTI DI CORRELAZIONE ----
% --- INTERPRETAZIONE DEL COEFFICIENTE DI CORRELAZIONE ---
% r > 0 : Correlazione POSITIVA (le variabili crescono insieme)
% r < 0 : Correlazione NEGATIVA (una cresce, l'altra decresce)
% r = 0 : ASSENZA di correlazione lineare
disp(" ")
disp("---- COEFFICIENTI DI CORRELAZIONE ----")
% coefficienti di correlazione:
corr_coeff1 = corr(X1_tr,Y_tr);
disp(" ")
disp("Coefficiente di correlazione con il genere")
disp(corr_coeff1)

corr_coeff2 = corr(X2_tr,Y_tr);

disp("Coefficiente di correlazione con l'età")
disp(corr_coeff2)

corr_coeff3 = corr(X3_tr, Y_tr);

disp("Coefficiente di correlazione con l'indice di massa corporea")
disp(corr_coeff3);

corr_coeff4 = corr(X4_tr, Y_tr);

disp("Coefficiente di correlazione con la glicemia a digiuno")
disp(corr_coeff4)

corr_coeff5 = corr(X5_tr, Y_tr);

disp("Coefficiente di correlazione con l'insulina")
disp(corr_coeff5)

corr_coeff6 = corr(X6_tr, Y_tr);


disp("Coefficiente di correlazione con il colesterolo totale")
disp(corr_coeff6);


corr_coeff7 = corr(X7_tr, Y_tr);


disp("Coefficiente di correlazione con la frequenza LTPA settimanale")
disp(corr_coeff7)

% ---- R^2 ----
disp(" ")
disp("---- R^2 ----")
R2_train = lm.Rsquared.Adjusted;
disp("R^2: ")
disp(R2_train)
if R2_train < 0.5
    disp("Il modello NON spiega adeguatamente i dati di training.");
else
    disp("Il modello spiega adeguatamente i dati di training.");
end

% ---- F-TEST ----
disp(" ")
disp("---- F-TEST ----")
anovaTrain = anova(lm,'summary');
Fstat_train = lm.ModelFitVsNullModel.Fstat;
fcrit95_train = finv(0.95,length(X1_tr),anovaTrain.DF(2));

disp("la F-test presenta il seguente valore: ")
disp(Fstat_train)
disp("la F a livello di significatività presenta il seguente valore: ")
disp(fcrit95_train)

if Fstat_train > fcrit95_train
    disp("Il modello è significativo sul training set.");
else
    disp("Il modello NON è significativo sul training set.");
end

% ---- p-value ----
disp(" ")
disp("---- p-value ----")
%studio il p-value
pValueTrain = anovaTrain.pValue(2);
disp("il pValue presenta il seguente valore: ");
disp(pValueTrain);


%il p-value del modello è molto basso rispetto al critico (0,05) il che
%indica che il modello è significativo per i dati

if pValueTrain < 0.005
 disp("Il modello potrebbe essere significativo per il nostro dataset");
else
   disp("Il modello non potrebbe essere significativo per il nostro dataset");
end

% ---- NORMALITÀ RESIDUI TRAIN ----
residui_train = lm.Residuals.Raw;
[h_tr,p_tr] = jbtest(residui_train);

if h_tr == 0
    disp("Il test di Jarque-Bera NON rifiuta l'ipotesi di normalità dei residui.");
    disp("Le inferenze statistiche sul modello possono considerarsi affidabili.");
else
    disp("Il test di Jarque-Bera rifiuta l'ipotesi di normalità dei residui.");
    disp("Le inferenze statistiche sul modello potrebbero non essere affidabili.");
end

% ================= TEST SET =================
test_data= readmatrix("test_dataset.csv");

Y_gt = test_data(:, 2);  % Assuming the second column corresponds to Glycohemoglobin
X1_te = test_data(:, 3); % Assuming the third column corresponds to Gender
X2_te = test_data(:, 4); % Assuming the fourth column corresponds to Age
X3_te = test_data(:, 5); % Assuming the fifth column corresponds to Body Mass Index
X4_te = test_data(:, 6); % Assuming the sixth column corresponds to Fasting Plasma Glucose
X5_te = test_data(:, 7); % Assuming the seventh column corresponds to Insulin
X6_te = test_data(:, 8); % Assuming the eighth column corresponds to Total Cholesterol
X7_te = test_data(:, 9); % Assuming the nineth column corresponds to LTPA Weekly frequency

% previsione sui dati di test
predictions = predict(lm, table(X1_te, X2_te, X3_te, X4_te, X5_te, X6_te, X7_te, 'VariableNames', {'Gender', 'Age', 'BMI', 'FastingGlucose', 'Insulin', 'Cholesterol', 'LTPA'}));

% Confronto Y_gd con il train_data
% Confronta le previsioni con i valori reali
figure;
scatter(Y_gt, predictions);
xlabel('Valori Reali di Glycohemoglobin');
ylabel('Previsioni');
title('Confronto tra Valori Reali e Previsioni');
grid on;
disp(" ")
disp("---- VALUTAZIONE DELLE PRESTAZIONI OUT-OF-SAMPLE ----")

% ---- RMSE ----
% Calcola l'errore quadratico medio (RMSE) delle previsioni
rmse = sqrt(mean((Y_gt - predictions).^2));
disp(" ")
disp("Errore quadratico medio (RMSE) delle previsioni:");
disp(rmse);

if rmse < std(Y_gt)
    disp("L'RMSE è accettabile rispetto alla variabilità dei dati di test.");
else
    disp("L'RMSE è elevato: il modello generalizza poco sui dati di test.");
end

% --- R^2 OUT-OF-SAMPLE ---

SS_res = sum((Y_gt - predictions).^2);          % Somma dei residui
SS_tot = sum((Y_gt - mean(Y_tr)).^2);            % Varianza totale

R2_out = 1 - (SS_res / SS_tot);
disp(" ")
disp("R^2 out-of-sample:");
disp(R2_out);

if R2_out > 0.5
    disp("Il modello mostra una buona capacità predittiva out-of-sample.");
elseif R2_out > 0
    disp("Il modello ha una capacità predittiva limitata out-of-sample.");
else
    disp("Il modello NON generalizza correttamente sui dati di test.");
end

% ---- CONCLUSIONE AUTOMATICA ----
if (R2_train > 0.5) && (R2_out > 0.5)
    disp("Il modello è affidabile sia in training che in test.");
elseif (R2_train > 0.5) && (R2_out <= 0.5)
    disp("Possibile overfitting: buono sul training, debole sul test.");
else
    disp("Il modello non è adeguato a descrivere il fenomeno.");
end