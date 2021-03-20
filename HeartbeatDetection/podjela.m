clear all
close all
clc

%% Ucitavanje podataka 

abnormal = load('ptbdb_abnormal1.txt')';
normal = load('ptbdb_normal1.txt')';
podaci = [normal abnormal];
[M, N] = size(podaci);

%% Obrada

y = podaci(188,:);
x = podaci(1:187,:);

%% Razdvajanje podataka

x0 = x(:,y==0);  % normalni
x1 = x(:,y==1);  % abnormalni
y0 = y(:,y==0);
y1 = y(:,y==1);

%% Permutovanje podataka

ind = randperm(length(x0));
num = round(0.80*length(ind));

x0Train = x0(:, ind(1 : num));
y0Train = y0(:,ind(1 : num));
x0Test=x0(:,ind(num+1:end));
y0Test=y0(:,ind(num+1:end));

ind = randperm(length(x1));
num = round(0.80*length(ind));

x1Train = x1(:, ind(1 : num));
y1Train = y1(:,ind(1 : num));
x1Test=x1(:,ind(num+1:end));
y1Test=y1(:,ind(num+1:end));

xTrain = [x0Train x1Train];
yTrain = [y0Train y1Train];

ind = randperm(length(xTrain));
xTrain = xTrain(:,ind);
yTrain = yTrain(:,ind);

xTest=[x0Test x1Test];
yTest=[y0Test y1Test];
ind = randperm(length(xTest));
xTest = xTest(:,ind);
yTest = yTest(:,ind);
%% Izvoz podataka

save('xTrain.mat', 'xTrain');
save('yTrain.mat', 'yTrain');
save('xTest.mat', 'xTest');
save('yTest.mat', 'yTest');
