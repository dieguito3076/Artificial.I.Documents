
Q = size(P, 2);
Z = [P; ones(1,Q)];

% clasificacion uno vs todos (10 neuronas)
T = [ones(1,500); -ones(9,500)];
for i=1:9
    T = [T [-ones(i,500); ones(1,500); -ones(9-i,500)]];
end
            
% Aprendizaje ADALINE basado en la minimización Ecm
R = 1/Q*Z*Z'; 
H = 1/Q*Z*T';
Xmin = pinv(R)*H; %R es singular por eso usar la pseudoinversa
W = Xmin(1:400,:)';
b = Xmin(401,:)';

% Verificación del Numero de Aciertos casi 90% de aciertos
for q=1:Q
    [a(q) iwin(q)] = max(W*P(:,q) + b);
end
y = [1*ones(1,500)];
for i=2:10
    y = [y [i*ones(1,500)]];
end
NumeroAciertos = sum(y==iwin)
PorcentajeAciertos = NumeroAciertos/5000*100

%Rutina aleatoria para ver imagen y el reconocimiento
figure, 
for veces=1:50
    q = round(4999*rand + 1);
    numeroreconocido = iwin(q)-1
    imshow(vec2mat(P(:,q),20)','InitialMagnification',300)
    pause
end



