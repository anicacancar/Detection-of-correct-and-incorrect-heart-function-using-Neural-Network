clear all
close all
clc

%% Ucitavanje podataka

load('xTrain.mat');
load('yTrain.mat');
load('xTest.mat');
load('yTest.mat');
xTrain = xTrain(:,:);
yTrain = yTrain(:,:);
xTest = xTest(:,:);
yTest = yTest(:,:);

%% Krosvalidacija

N_train=length(xTrain);
best_acc=0;
best_f1=0;
best_structure=[3 3];
best_trainFcn = 'poslin';
best_reg= 0.2;
 best_c0_weight = 0;

for structure = {[10 10 10], [30 30 30 30],[20 20 20 20 20]}
    for trainFcn = {'tansig','poslin','logsig'}
        for reg = {0.1,0.3,0.5}
           for c0_weight = {1.5,2,4,5}
                net = patternnet(structure{1});
                 for i=1:length(structure)
                        net.layers{i}.transferFcn = trainFcn{1};
                 end
                net.performParam.regularization = reg{1};

                net.divideFcn = 'divideind';
                net.divideParam.trainInd = [1:round(0.7*N_train)];
                net.divideParam.valInd = [round(0.7*N_train)+1:N_train];
                net.divideParam.testInd = [];

                % Dobavljanje validacionih podataka
                X_val = xTrain(:, net.divideParam.valInd);
                Y_val = yTrain(:, net.divideParam.valInd);

                net.trainParam.max_fail = 37;
                net.trainParam.goal = 1e-6;

                net.trainParam.epochs = 2000;
                net.trainParam.min_grad = 0;
                net.trainparam.showWindow = false;

                w = ones(1,length(xTrain));
                w(yTrain==1) = c0_weight{1};
                [net,tr] = train(net, xTrain, yTrain,[],[],w);

                Y_val_pred = sim(net,X_val);

                [c,cm_val,ind,per] = confusion(Y_val,Y_val_pred);
                cm_val = cm_val';
                recall = cm_val(2,2)/(cm_val(1,2)+cm_val(2,2));
                precision = cm_val(2,2)/(cm_val(2,1)+cm_val(2,2));
                f1=2*precision*recall/(precision+recall);
                acc=(cm_val(1,1)+cm_val(2,2))/sum(sum(cm_val));

                if f1>best_f1
                    best_acc=acc;
                    best_f1=f1;
                    best_structure=structure{1};
                    best_trainFcn=trainFcn{1};
                    best_reg=reg{1};
                    best_epoch=tr.best_epoch;
                    best_c0_weight = c0_weight;

                end
            end
        end
    end
end

%%
disp(" ")
disp("Rezultat krosvalidacije: ")
disp("Najbolja struktura:  " + num2str(best_structure))
disp("Najbolja aktivaciona funkcija:  " + string(best_trainFcn))
disp("Najbolji koeficijent regularizacije:  " + string(best_reg))
disp("Najbolja duzina treniranja sa ovim parametrima:  " + string(best_epoch))
disp(" ")
disp("Najbolji accuracy:  " + string(best_acc))
disp("Najbolji f1:  " + string(best_f1))
disp(" ")
disp(" ")

%% Treniranje neuralne mreze

net = patternnet(best_structure);
for i=1:length(best_structure)
    net.layers{i}.transferFcn = best_trainFcn;
end
            

            net.performParam.regularization = best_reg;
            
            net.divideFcn = 'divideind';
            net.divideParam.trainInd = [1:length(xTrain)];
            net.divideParam.valInd = [];
            net.divideParam.testInd = [];
            
            % Dobavljanje validacionih podataka
            X_val = xTrain(:, net.divideParam.valInd);
            Y_val = yTrain(:, net.divideParam.valInd);
            
            net.trainParam.max_fail = 37;
            net.trainParam.goal = 1e-6;
            
            net.trainParam.epochs = best_epoch;
            net.trainParam.min_grad = 0;
            net.trainparam.showWindow = true;
            
            w = ones(1,length(xTrain));
            w(yTrain==1) = best_c0_weight{1};
            [net,tr] = train(net, xTrain, yTrain,[],[],w);
            

%% Trening skup

close all

Y_train_net=sim(net,xTrain);

[c,cm,ind,per]=confusion(yTrain,Y_train_net);
cm=cm';

figure
plotconfusion(yTrain,Y_train_net);
title('Trening')
%% Test skup

Y_test_net=sim(net,xTest);
% [c,cm,ind,per]=confusion(yTest,Y_test_net);
% cm=cm';
figure
plotconfusion(yTest,Y_test_net);
title('Test')
