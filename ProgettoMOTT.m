%% Svolgimento progetto 8 Modelli e Metodi di Ottimizzazione statistica di Paolo Ardolino, Giovanni Cafarelli, Nicola Inchingolo
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
% Crea la tabella passando ogni colonna individualmente
tbl = table(X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, X6_tr, X7_tr, Y_tr, 'VariableNames', {'Gender', 'Age', 'BMI', 'FastingGlucose', 'Insulin', 'Cholesterol', 'LTPA', 'Glycohemoglobin'});

% Esegui il fit del modello usando i nuovi nomi (senza spazi per evitare errori)
lm = fitlm(tbl, 'Glycohemoglobin ~ Age + BMI + Gender + FastingGlucose + Insulin + Cholesterol + LTPA')

plot(lm)
% valutazione bontà del modello:
% valuto R^2 e p-value
% Ftest

% --- INTERPRETAZIONE DEL COEFFICIENTE DI CORRELAZIONE ---
% r > 0 : Correlazione POSITIVA (le variabili crescono insieme)
% r < 0 : Correlazione NEGATIVA (una cresce, l'altra decresce)
% r = 0 : ASSENZA di correlazione lineare

% coefficienti di correlazione:
corr_coeff1 = corr(X1_tr,Y_tr);

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

% compreso tra -1 e 1. 

test_data= readmatrix("train_dataset.csv");

Y_gt = test_data(:, 2);  % Assuming the second column corresponds to Glycohemoglobin
X1_te = test_data(:, 3); % Assuming the third column corresponds to Gender
X2_te = test_data(:, 4); % Assuming the fourth column corresponds to Age
X3_te = test_data(:, 5); % Assuming the fifth column corresponds to Body Mass Index
X4_te = test_data(:, 6); % Assuming the sixth column corresponds to Fasting Plasma Glucose
X5_te = test_data(:, 7); % Assuming the seventh column corresponds to Insulin
X6_te = test_data(:, 8); % Assuming the eighth column corresponds to Total Cholesterol
X7_te = test_data(:, 9); % Assuming the nineth column corresponds to LTPA Weekly frequency

% Esegui la previsione sui dati di test
predictions = predict(lm, table(X1_te, X2_te, X3_te, X4_te, X5_te, X6_te, X7_te, 'VariableNames', {'Gender', 'Age', 'BMI', 'FastingGlucose', 'Insulin', 'Cholesterol', 'LTPA'}));

% Confronto Y_gd con il train_data
% Confronta le previsioni con i valori reali
figure;
scatter(Y_gt, predictions);
xlabel('Valori Reali di Glycohemoglobin');
ylabel('Previsioni');
title('Confronto tra Valori Reali e Previsioni');
grid on;

% Calcola l'errore quadratico medio (RMSE) delle previsioni
rmse = sqrt(mean((Y_gt - predictions).^2));
disp("Errore quadratico medio (RMSE) delle previsioni:");
disp(rmse);

