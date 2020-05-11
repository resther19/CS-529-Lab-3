%%%%%%%% Data Representation Figures %%%%%%%%
%%% Created by Esther Rodriguez
%%% April 2020

clc
close all
clear

%% Read audio files to create one figure for each class
[y0,~] = audioread('train/00994050.mp3');
[y1,~] = audioread('train/00918098.mp3');
[y2,~] = audioread('train/00977052.mp3');
[y3,~] = audioread('train/01027602.mp3');
[y4,~] = audioread('train/00933942.mp3');
[y5,~] = audioread('train/00919080.mp3');

%% Time Series figures
t0 = linspace(0,length(y0)/44100,length(y0))';
t1 = linspace(0,length(y1)/44100,length(y1))';
t2 = linspace(0,length(y2)/44100,length(y2))';
t3 = linspace(0,length(y3)/44100,length(y3))';
t4 = linspace(0,length(y4)/44100,length(y4))';
t5 = linspace(0,length(y5)/44100,length(y5))';

figure('color','w')
sgtitle('Time Series','Fontweight','bold')

subplot(2,3,1)
plot(t0,y0(:,1))
title('Rock')
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,2)
plot(t1,y1(:,1))
title('Pop')
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,3)
plot(t2,y2(:,1))
title('Folk')
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,4)
plot(t3,y3(:,1))
title('Instrumental')
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,5)
plot(t4,y4(:,1))
title('Electronic')
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,6)
plot(t5,y5(:,1))
title('Hip-Hop')
ax = gca;
ax.FontWeight = 'bold';

%% MFCC spectrogram figures
fs = 44100;

figure('color','w')
sgtitle('Mel Frequency Spectrogram','Fontweight','bold')

subplot(2,3,1)
melSpectrogram(y0,fs);
title('Rock')
colorbar off
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,2)
melSpectrogram(y1,fs);
title('Pop')
colorbar off
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,3)
melSpectrogram(y2,fs);
title('Folk')
colorbar off
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,4)
melSpectrogram(y3,fs);
title('Instrumental')
colorbar off
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,5)
melSpectrogram(y4,fs);
title('Electronic')
colorbar off
ax = gca;
ax.FontWeight = 'bold';

subplot(2,3,6)
melSpectrogram(y5,fs);
title('Hip-Hop')
colorbar off
ax = gca;
ax.FontWeight = 'bold';


%% Wavelets Scattergrams figures
sf = waveletScattering('SignalLength',2^19,'SamplingFrequency',22050,...
    'InvarianceScale',0.5);

[S0,U0] = scatteringTransform(sf,y0(1:2^19,1));
[S1,U1] = scatteringTransform(sf,y1(1:2^19,1));
[S2,U2] = scatteringTransform(sf,y2(1:2^19,1));
[S3,U3] = scatteringTransform(sf,y3(1:2^19,1));
[S4,U4] = scatteringTransform(sf,y4(1:2^19,1));
[S5,U5] = scatteringTransform(sf,y5(1:2^19,1));


figure('color','w')
scattergram(sf,U0,'FilterBank',1)
title('Scattergram - Filter Bank 1 - Rock')
ax = gca;
ax.FontWeight = 'bold';

figure('color','w')
scattergram(sf,U1,'FilterBank',1)
title('Scattergram - Filter Bank 1 - Pop')
ax = gca;
ax.FontWeight = 'bold';

figure('color','w')
scattergram(sf,U2,'FilterBank',1)
title('Scattergram - Filter Bank 1 - Folk')
ax = gca;
ax.FontWeight = 'bold';


figure('color','w')
scattergram(sf,U3,'FilterBank',1)
title('Scattergram - Filter Bank 1 - Instrumental')
ax = gca;
ax.FontWeight = 'bold';

figure('color','w')
scattergram(sf,U4,'FilterBank',1)
title('Scattergram - Filter Bank 1 - Electronic')
ax = gca;
ax.FontWeight = 'bold';


figure('color','w')
scattergram(sf,U5,'FilterBank',1)
title('Scattergram - Filter Bank 1 - Hip-Hop')
ax = gca;
ax.FontWeight = 'bold';


figure('color','w')
scattergram(sf,U0,'FilterBank',2)
title('Scattergram - Filter Bank 2 - Rock')
ax = gca;
ax.FontWeight = 'bold';


figure('color','w')
scattergram(sf,U1,'FilterBank',2)
title('Scattergram - Filter Bank 2 - Pop')
ax = gca;
ax.FontWeight = 'bold';

figure('color','w')
scattergram(sf,U2,'FilterBank',2)
title('Scattergram - Filter Bank 2 - Folk')
ax = gca;
ax.FontWeight = 'bold';


figure('color','w')
scattergram(sf,U3,'FilterBank',2)
title('Scattergram - Filter Bank 2 - Instrumental')
ax = gca;
ax.FontWeight = 'bold';


figure('color','w')
scattergram(sf,U4,'FilterBank',2)
title('Scattergram - Filter Bank 2 - Electronic')
ax = gca;
ax.FontWeight = 'bold';


figure('color','w')
scattergram(sf,U5,'FilterBank',2)
title('Scattergram - Filter Bank 2 - Hip-Hop')
ax = gca;
ax.FontWeight = 'bold';



