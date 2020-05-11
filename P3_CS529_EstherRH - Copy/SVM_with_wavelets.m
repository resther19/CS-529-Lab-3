%%%%% References https://www.mathworks.com/help/wavelet/examples/music-genre-classification-using-wavelet-scattering.html
%%% Adapted and Modified by Esther Rodriguez
%%% April 2020

clear
clc
close all
%% Scaling Function and Coarsest-Scale Wavelet First Filter Bank figure

sf = waveletScattering('SignalLength',2^19,'SamplingFrequency',22050,...
    'InvarianceScale',0.5);

[fb,f,filterparams] = filterbank(sf);
phi = ifftshift(ifft(fb{1}.phift));
psiL1 = ifftshift(ifft(fb{2}.psift(:,end)));
dt = 1/22050;
time = -2^18*dt:dt:2^18*dt-dt;
scalplt = plot(time,phi,'linewidth',1.5);
hold on
grid on
ylimits = [-3e-4 3e-4];
ylim(ylimits);
plot([-0.25 -0.25],ylimits,'k--');
plot([0.25 0.25],ylimits,'k--');
xlim([-0.6 0.6]);
xlabel('Seconds'); ylabel('Amplitude');
wavplt = plot(time,[real(psiL1) imag(psiL1)]);
legend([scalplt wavplt(1) wavplt(2)],{'Scaling Function','Wavelet-Real Part','Wavelet-Imaginary Part'});
title({'Scaling Function';'Coarsest-Scale Wavelet First Filter Bank'})
hold off

%% Read data from the train folder and slipt it into 80-20 for testing

location = fullfile('train');
ads = audioDatastore(location,'LabelSource','none');
A = readmatrix('train_mat.csv');
ads.Labels = A;
countEachLabel(ads)
[adsTrain,adsTest] = splitEachLabel(ads,0.8);
countEachLabel(adsTrain)
countEachLabel(adsTest)
Ttrain = tall(adsTrain);
Ttest = tall(adsTest);

%% Wavelet scattergram to extract features

scatteringTrain = cellfun(@(x)helperscatfeatures(x,sf),Ttrain,'UniformOutput',false);
scatteringTest = cellfun(@(x)helperscatfeatures(x,sf),Ttest,'UniformOutput',false);
addAttachedFiles(gcp(),'helperscatfeatures')

TrainFeatures = gather(scatteringTrain);
TrainFeatures = cell2mat(TrainFeatures);

TestFeatures = gather(scatteringTest);
TestFeatures = cell2mat(TestFeatures);

%% Bin data into scattering time windows corresponding to the wavelet scattering. There are 32 such windows.

numTimeWindows = 32;
trainLabels = adsTrain.Labels;
numTrainSignals = numel(trainLabels);
trainLabels = repmat(trainLabels,1,numTimeWindows);
trainLabels = reshape(trainLabels',numTrainSignals*numTimeWindows,1);

testLabels = adsTest.Labels;
numTestSignals = numel(testLabels);
testLabels = repmat(testLabels,1,numTimeWindows);
testLabels = reshape(testLabels',numTestSignals*numTimeWindows,1);


%% SVM Model

template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
Classes = {'0','1','2','3','4','5'};
classificationSVM = fitcecoc(...
    TrainFeatures, ...
    trainLabels, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', categorical(Classes));                 

%% Predicts the lables for the 20% split

predLabels = predict(classificationSVM,TestFeatures);

% Make categorical arrays if the labels are not already categorical
PL = categorical(predLabels);
origLabels = categorical(adsTest.Labels);
% Expects both predLabels and origLabels to be categorical vectors
Npred = numel(PL);
Norig = numel(origLabels);
Nwin = Npred/Norig;
PL = reshape(PL,Nwin,Norig);
TestCounts = countcats(PL);
[mxcount,idx] = max(TestCounts);
classes = categorical(Classes)
ClassVotes = classes(idx);
% Check for any ties in the maximum values and ensure they are marked as
% error if the mode occurs more than once
modecnt = modecount(TestCounts,mxcount);
ClassVotes(modecnt>1) = categorical({'NoUniqueMode'});
TestVotes = ClassVotes(:);

%Accuracy for the 20% split
TestVotes(TestVotes=='NoUniqueMode') = categorical(randi(6,sum(TestVotes=='NoUniqueMode'),1) - 1);
testAccuracy = sum(eq(TestVotes,categorical(adsTest.Labels)))/numTestSignals*100


%% Confusion matrix for the SVM model

figure('color','w')
C = confusionmat(TestVotes,categorical(adsTest.Labels));
C = C(1:6,1:6);
categorical(adsTest.Labels)
CM = confusionchart(C',categorical(Classes));
CM.FontSize = 14;
title('Confusion Matrix for SVM')


%% Confidence interval for the SVM model

ciradius_SVM = 1.96*sqrt(0.5*0.5/(0.2*2400));


%% Confusion matrix for the python trained CNN (figure generated here for consistency)

%Read data from the output file of the predictions for test set done by the CNN trained in python
T = readmatrix('80_20_predictions.csv');
T = T(2:end,:);
%Finds the files that corresponds to the 80-20 split in the SVM Model
y_actual = [T(321:400,2), T(721:800,2), T(1121:1200,2), T(1521:1600,2), T(1921:2000,2), T(2321:2400,2)];
y_actual = y_actual(:);
y_pred = [T(321:400,9), T(721:800,9), T(1121:1200,9), T(1521:1600,9), T(1921:2000,9), T(2321:2400,9)];
y_pred = y_pred(:);

figure('color','w')
Cn = confusionmat(categorical(y_pred),categorical(y_actual));
categorical(adsTest.Labels);
CMn = confusionchart(Cn',categorical(Classes));
CMn.FontSize = 14;
title('Confusion Matrix for CNN')

%% Confidence interval for the python trined CNN (calculated here for consistency)

ciradius_CNN = 1.96*sqrt(0.883*(1-0.883)/(0.2*2400));
