%% Svolgimento progetto 8 Modelli e Metodi di Ottimizzazione statistica di Paolo Ardolino, Giovanni Cafarelli, Nicola Inchingolo
data = readmatrix("Dataset_Diabete_Completo.csv");

Y = data(:, 2);  % Assuming the second column corresponds to Glycohemoglobin
X1 = data(:, 3); % Assuming the third column corresponds to Gender
X2 = data(:, 4); % Assuming the fourth column corresponds to Age
X3 = data(:, 5); % Assuming the fifth column corresponds to Body Mass Index
X4 = data(:, 6); % Assuming the sixth column corresponds to Fasting Plasma Glucose
X5 = data(:, 7); % Assuming the seventh column corresponds to Insulin
X6 = data(:, 8); % Assuming the eighth column corresponds to Total Cholesterol
X7 = data(:, 9); % Assuming the nineth column corresponds to LTPA Weekly frequency


% regressione lineare multipla
% Crea la tabella passando ogni colonna individualmente
tbl = table(X1, X2, X3, X4, X5, X6, X7, Y, 'VariableNames', {'Gender', 'Age', 'BMI', 'FastingGlucose', 'Insulin', 'Cholesterol', 'LTPA', 'Glycohemoglobin'});

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
corr_coeff1 = corr(X1,Y);

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

% compreso tra -1 e 1. 

disp(lm.Rsquared)

R_squared= lm.Rsquared.Adjusted

if  R_squared<0.5
    disp("Il modello non spiega sufficientemente la variabilità dei dati.");
else
    disp("Il modello spiega adeguatamente la variabilità dei dati.");
end

% Il comando corr(x,y) restituisce anche il p-value
% Se p < 0.05 : La correlazione è STATISTICAMENTE SIGNIFICATIVA
%               (C'è meno del 5% di probabilità che sia un caso)
% Se p > 0.05 : La correlazione NON è significativa 
%               (Il legame osservato potrebbe essere dovuto al caso)

anova_Res= anova(lm,'summary')

p = anova_Res.pValue;






