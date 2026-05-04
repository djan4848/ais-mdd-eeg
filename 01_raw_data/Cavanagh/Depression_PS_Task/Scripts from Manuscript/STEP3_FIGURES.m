%%
clear all; clc
rootpath='Y:\EEG_Data\PL Cort Depression\FOR UPLOAD\';  addpath(rootpath);  cd(rootpath);
datapath='Y:\EEG_Data\PL Cort Depression\Processed Data\'; 

load([rootpath,'NScan_Chanlocs_60.mat'],'NScan_Chanlocs_60');

[NUM,TXT,RAW]=xlsread('Data_4_Import.xlsx');

Filz=dir([datapath,'*_TF.mat']);  

% Preallocate
for chani=1:2
    MEGA_POWER{chani}=NaN(length(Filz),50,751,2);
    MEGA_ITPC{chani}=NaN(length(Filz),50,751,2);
    MEGA_PWR_CORR{chani}=NaN(length(Filz),50,751,2);
end
MEGA_ERP=NaN(length(Filz),60,751,2);
MEGA_ERP_CORR=NaN(length(Filz),60,751,2);
MEGA_ERP_PEs=NaN(length(Filz),60,751,2,2);
MEGA_ERPs_PEs_cts=NaN(length(Filz),2,2);
MEGA_LLE=NaN(length(Filz),3);
MEGA_Params=NaN(length(Filz),3);

for si=1:length(Filz)  
    
    clc; disp(Filz(si).name);
    load([datapath,Filz(si).name]);
    
    % Invert -PE to abs(-PE)
    ERP_CORR(:,:,1)=-1.*ERP_CORR(:,:,1);
    for chani=1:2
        PWR_CORR{chani}=-1.*PWR_CORR{chani}(:,:,:,1);
    end
    
    for chani=1:2
        MEGA_POWER{chani}(si,:,:,:)=POWER{chani};
        MEGA_ITPC{chani}(si,:,:,:)=ITPC{chani};
        MEGA_PWR_CORR{chani}(si,:,:,:)=PWR_CORR{chani};
    end
    MEGA_ERP(si,:,:,:)=ERPs;
    MEGA_ERP_CORR(si,:,:,:)=ERP_CORR;
    
    MEGA_ERPs_PEs_cts(si,:,:)=ERPs_PEs_cts;
    MEGA_ERP_PEs(si,:,:,:,:)=ERPs_PEs;
    
    for mi=1:size(MODEL,2)
        MEGA_LLE(si,mi)=MODEL{mi}.LLE;
    end
    
    % Params from winning model
    MEGA_Params(si,:)=MODEL{end}.Params; 
    
    clear POWER ITPC ERPs PWR_CORR  ERP_CORR ID_MATRIX_V2 MODEL ERPs_PEs_cts ERPs_PEs;
   
end


% Here are the highly symptomatic participants grouped by SCID

AllDep=NUM(NUM(:,7)>8,2);
MDD=sum(AllDep==1);
Past=sum(AllDep==2);
NotMDD=sum(AllDep==50);
NoInt=sum(AllDep==99);
% figure; pie([MDD,Past,NotMDD,NoInt],{'MDD','Past Hx','Not MDD','No Interview'});

% Look at demographics and performance
CTL=NUM(:,7)<8;     
DEP=NUM(:,7)>8;   

TheseData=NUM;
% Blank out 544, who had unstable BDI between assessment and intake
TheseData([logical(1-logical(double(CTL)+double(DEP)))],:)=NaN;


figure;
TOPLOT=[7,11,12,13,15,16,17,18,6];
subplot(3,3,1); hold on
bar(1-.2,sum(TheseData(CTL,5)==1),.3,'w');  % 1=F, 2=M
bar(1+.2,sum(TheseData(CTL,5)==2),.3,'w');  text(.5,10,[num2str(sum(TheseData(CTL,5)==1)./sum(CTL)),' %F']);
bar(2-.2,sum(TheseData(DEP,5)==1),.3,'w');
bar(2+.2,sum(TheseData(DEP,5)==2),.3,'w');  text(1.5,15,[num2str(sum(TheseData(DEP,5)==1)./sum(DEP)),' %F']);
set(gca,'xlim',[0 3],'xtick',[1:1:2],'xticklabel',{'CTL','DEP'});
for vari=1:8
    subplot(3,3,1+vari); hold on
    boxplot(TheseData(:,TOPLOT(vari)),DEP); title(TXT{1,TOPLOT(vari)});
end

% For Table
clear P;
for vari=1:9  % add age at end
    TABLE_mean(vari,1)=nanmean( TheseData(CTL,TOPLOT(vari)) );
    TABLE_mean(vari,2)=nanmean( TheseData(DEP,TOPLOT(vari)) );
    TABLE_std(vari,1)=nanstd( TheseData(CTL,TOPLOT(vari)) );
    TABLE_std(vari,2)=nanstd( TheseData(DEP,TOPLOT(vari)) );
    [~,P(vari),~,STATS(vari)]=ttest2( TheseData(CTL,TOPLOT(vari)) , TheseData(DEP,TOPLOT(vari)));
    TABLE_TXT{vari}=TXT{1,TOPLOT(vari)};
end

% For Table
for vari=1:3   
    TABLE_mean_model(vari,1)=nanmean( MEGA_Params(CTL,vari) );
    TABLE_mean_model(vari,2)=nanmean( MEGA_Params(DEP,vari) );
    TABLE_std_model(vari,1)=nanstd( MEGA_Params(CTL,vari) );
    TABLE_std_model(vari,2)=nanstd( MEGA_Params(DEP,vari) );
    [~,P_model(vari),~,STATS_model(vari)]=ttest2( MEGA_Params(CTL,vari) , MEGA_Params(DEP,vari));
end

% Model Vars
PseudoR2(:,1)=( MEGA_LLE(:,1)-MEGA_LLE(:,2) ) ./ MEGA_LLE(:,1);
PseudoR2(:,2)=( MEGA_LLE(:,1)-MEGA_LLE(:,3) ) ./ MEGA_LLE(:,1);
% PseudoR2(:,3)=( MEGA_LLE(:,1)-MEGA_LLE(:,4) ) ./ MEGA_LLE(:,1);
for mi=1:size(MEGA_Params,2)
    AIC(:,mi)=2*mi-1 - 2*(-MEGA_LLE(:,mi));
    
    PARAM(1,mi)=median(MEGA_Params(CTL,mi));
    PARAM(2,mi)=median(MEGA_Params(DEP,mi));
    PARAM(3,mi)=iqr(MEGA_Params(CTL,mi));
    PARAM(4,mi)=iqr(MEGA_Params(DEP,mi));
     [rsP(mi),rsH(mi),rsSTATS(mi)]=ranksum(MEGA_Params(CTL,mi),MEGA_Params(DEP,mi));
end

figure; 
subplot(1,3,1); hold on; boxplot(MEGA_LLE); title('LLE');
subplot(1,3,2); hold on; boxplot(PseudoR2); title('PseudoR^2');
subplot(1,3,3); hold on; boxplot(AIC); title('AIC');

% For Report
mean(MEGA_LLE)
mean(PseudoR2)
mean(AIC)

figure; hold on
bar(1,PARAM(1,1),'w');
bar(2,PARAM(2,1),'w');
errorbar(1,PARAM(1,1),PARAM(3,1)./sqrt(sum(CTL)),'k.');
errorbar(2,PARAM(2,1),PARAM(4,1)./sqrt(sum(DEP)),'k.');
set(gca,'xlim',[0 3],'xtick',1:1:2);

% Likelihood ratio test
LLEdiff=2*(mean(MEGA_LLE(:,2)-MEGA_LLE(:,3)))
LLEdiff_cdf=chi2cdf(LLEdiff,1); 
LLEdiff_p=1-LLEdiff_cdf



%%
% BDI sub-scales from Vanhuele et al., 2008
    % BDI_cog is 2,3,6,8,9,14
    % BDI_aff is 4,10,12
    % BDI_som is 16,17,18,19,20,21
TheseData(:,37)= ( (TheseData(:,8)*6) + (TheseData(:,9)*3) )  /9;  % BDI cog + aff


% % % % ##################### Remove shared variance % ##################### % #####################
% % % clear B BINT R;
% % % [B,BINT,R] = regress(  TheseData(DEP,37)  , [ones(sum(DEP),1) TheseData(DEP,11) ] );
% % % TheseData(DEP,38) = R;
% % % clear B BINT R;
% % % [B,BINT,R] = regress(  TheseData(DEP,11)  , [ones(sum(DEP),1) TheseData(DEP,37) ] );
% % % TheseData(DEP,39) = R;
% % % DEPVAR=38;   
% % % ANXVAR=39;   
% % % figure; hold on
% % % scatter(TheseData(DEP,11) ,TheseData(DEP,37) ,'b'); lsline
% % % scatter(TheseData(DEP,39) ,TheseData(DEP,37) ,'r'); lsline
% % % scatter(TheseData(DEP,38) ,TheseData(DEP,11) ,'k'); lsline
% % % % #####################  OR don't   % #####################  % #####################
DEPVAR=37;  % 7=BDI, 8=BDI_cog, 9=BDI_aff, 10=BDI_som,  37=BDI_cog & aff
ANXVAR=11;  % TAI
% % % % ##################### % ##################### % ##################### % #####################

NOGOVAR=18;
GOVAR=17;

% -------------- CONDI DIFFS
% mengz_JFC(r1, r2, r12, n) compares two correlations r1 and r2:
% r1: correlation between X and Y
% r2: correlation between X and Z
% r12: correlation between Y and Z
% n: number of observations used to compute correlations
[r12,p12]=corr(TheseData(DEP,DEPVAR),TheseData(DEP,ANXVAR),'rows','complete','type','Spearman')
n=length(TheseData(DEP,ANXVAR));

%% PERFORMANCE
GROUP=DEP;
V1=TheseData(GROUP,ANXVAR);    % Main correlate for beh (anx)
V1_C=TheseData(GROUP,DEPVAR);  % This one is the contrast (dep)

figure; 

subplot(4,4,[1,2,5,6]); hold on
V2=TheseData(GROUP,GOVAR)-TheseData(GROUP,NOGOVAR);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'k'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[30 70],'ylim',[-1 1]);
title('Go-NoGo')

subplot(4,4,9); hold on
V2=TheseData(GROUP,NOGOVAR);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'r'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[30 70],'ylim',[0 1]);
title('NoGo')
% ---
subplot(4,4,10);
V2=TheseData(GROUP,GOVAR);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'g'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[30 70],'ylim',[0 1]);
title('Go')

% ---% ---% ---% ---% ---% ---% ---% ---% ---% ---
subplot(4,4,[3,4,7,8]);  hold on
V2=TheseData(GROUP,30)-TheseData(GROUP,29);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'k'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[30 70],'ylim',[-1500 1500]);
title('Go-NoGo RT')

subplot(4,4,11);  hold on
V2=TheseData(GROUP,30);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'r'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[30 70],'ylim',[0 3500]);
title('NoGo RT')
% ---------
subplot(4,4,12);  hold on
V2=TheseData(GROUP,29);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'g'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[30 70],'ylim',[0 3500]);
title('Go RT')

% Models   - gain, loss, beta
[r_model,p_model]=corr(MEGA_Params(DEP,:),TheseData(DEP,ANXVAR),'rows','complete','type','Spearman')
[r_model,p_model]=corr(MEGA_Params(DEP,:),TheseData(DEP,DEPVAR),'rows','complete','type','Spearman')

%% ERPs and Topos

chani=find(strcmpi('FCz',{NScan_Chanlocs_60.labels}));

tx2disp=-500:2:1000;
rewp1=find(tx2disp==250);
rewp2=find(tx2disp==350);
pun_n2=find(tx2disp==276);
pun_p3=find(tx2disp==376);

PANELS=[[1:4,11:14];[7:10,17:20];[21:24,31:34];[27:30,37:40]];
CORR_PANELS=[[5,15];[6,16];[25,35];[26,36]];
COLSHAPE={'ro','rd';'go','gd';'ro','rd';'go','gd'};

TITLES={'Punishment','Reward','-PE Corr','+PE Corr'};
figure;
for typei=1:4
    if      typei==1,  data=MEGA_ERP;     YLIM=[-10 20];  idx=1;  SCATTERYLIM=[-10 30];
        var=squeeze( MEGA_ERP(:,chani,pun_p3,idx)-MEGA_ERP(:,chani,pun_n2,idx) ); t1=276; t2=376;
    elseif  typei==2,  data=MEGA_ERP;     YLIM=[-10 20];  idx=2;  SCATTERYLIM=[-10 30];
        var=squeeze( mean(MEGA_ERP(:,chani,rewp1:rewp2,idx),3)  );      t1=250; t2=350;
    elseif  typei==3,  data=MEGA_ERP_CORR;YLIM=[-.1 .15]; idx=1;  SCATTERYLIM=[-.5 .5];
        var=squeeze( mean(MEGA_ERP_CORR(:,chani,pun_n2:pun_p3,idx),3)  ); t1=276; t2=376;
    elseif  typei==4,  data=MEGA_ERP_CORR;YLIM=[-.1 .15]; idx=2;  SCATTERYLIM=[-.5 .5];
        var=squeeze( mean(MEGA_ERP_CORR(:,chani,rewp1:rewp2,idx),3)  ); t1=250; t2=350;
    end
    
    % ^^^^^^^^^^^^^
    subplot(4,10,PANELS(typei,:)); hold on
    if typei==1, 
        plot(tx2disp,squeeze(mean(data(CTL,chani,:,idx),1)),'b','linewidth',2);
        plot(tx2disp,squeeze(mean(data(DEP,chani,:,idx),1)),'k','linewidth',2);
        legend({'CTL','DEP'},'Location','NorthWest');
    end
    shadedErrorBar(tx2disp,squeeze(mean(data(CTL,chani,:,idx),1)),squeeze(std(data(CTL,chani,:,idx),1))./sqrt(sum(CTL)),'b');
    shadedErrorBar(tx2disp,squeeze(mean(data(DEP,chani,:,idx),1)),squeeze(std(data(DEP,chani,:,idx),1))./sqrt(sum(DEP)),'k');
    plot(tx2disp,squeeze(mean(data(CTL,chani,:,idx),1)),'b','linewidth',2);
    plot(tx2disp,squeeze(mean(data(DEP,chani,:,idx),1)),'k','linewidth',2);
    set(gca,'ylim',YLIM); plot([0 0],YLIM,'k:');  plot([-500 1000],[0 0],'k:');
    plot([t1 t1],YLIM,'m'); plot([t2 t2],YLIM,'m');
    title(TITLES{typei});
    
    % ^^^^^^^^^^^^^
    [H,P,CI,STATS]=ttest2(data(CTL,chani,:,idx),data(DEP,chani,:,idx));
    P=squeeze(P); P(P>=.05)=NaN;  P(P<.05)=1; 
    plot(tx2disp,P.*9*(YLIM(end)/10),'y','linewidth',2);
    if typei==3 || typei==4,
        [H,P,CI,STATS]=ttest(data(:,chani,:,idx));
        P=squeeze(P); P(P>=.05)=NaN;  P(P<.05)=1;
        plot(tx2disp,P.*8*(YLIM(end)/10),'c','linewidth',2);
    end
    [H,P_ERPs(typei),CI,STATS_ERPs(typei)]=ttest2(var(CTL),var(DEP));
    

    % ^^^^^^^^^^^^^
    subplot(4,10,CORR_PANELS(typei,1)); hold on
    scatter(TheseData(DEP,ANXVAR),var(DEP),COLSHAPE{typei,1}); lsline
    [r_a(typei),p_a(typei)]=corr(TheseData(DEP,ANXVAR),var(DEP),'rows','complete','type','Spearman');
    text(.1,.2,['rho=',num2str(r_a(typei))],'sc');      text(.1,.1,['p=',num2str(p_a(typei))],'sc');
%    set(gca,'xlim',[30 70],'ylim',SCATTERYLIM);
    subplot(4,10,CORR_PANELS(typei,2)); hold on
    scatter(TheseData(DEP,DEPVAR),var(DEP),COLSHAPE{typei,2}); lsline
    [r_d(typei),p_d(typei)]=corr(TheseData(DEP,DEPVAR),var(DEP),'rows','complete','type','Spearman');
    text(.1,.2,['rho=',num2str(r_d(typei))],'sc');      text(.1,.1,['p=',num2str(p_d(typei))],'sc');
%    set(gca,'xlim',[0 2],'ylim',SCATTERYLIM);
    % ^^^^^^^^^^^^^
    clear data idx var P STATS ylim t1 t2;
    
end

% -------------- CONDI DIFFS

% Diff in ERP ROIs between groups
STATS_ERPs(:).tstat
P_ERPs(:)

% Diff between correlations

% -------------- PUN & DEP vs. REW & DEP
[menghyp(1),mengp(1),mengzscore(1)] = mengz_JFC(r_d(1),r_d(2),r12,n);

% -------------- REW & ANXIETY vs. REW & DEP
[menghyp(2),mengp(2),mengzscore(2)] = mengz_JFC(r_a(2),r_d(2),r12,n);

% -------------- PUN & ANXIETY vs. REW & DEP  
[menghyp(3),mengp(3),mengzscore(3)] = mengz_JFC(r_a(1),r_d(2),r12,n);

% --------   PE   ------ PUN & ANXIETY vs. PUN & DEP
[menghyp(4),mengp(4),mengzscore(4)] = mengz_JFC(r_a(1+2),r_d(1+2),r12,n);

% --------   PE   ------ PUN & ANXIETY vs. REW & ANXIETY
[menghyp(5),mengp(5),mengzscore(5)] = mengz_JFC(r_a(1+2),r_a(2+2),r12,n);

% --------   PE   ------ PUN & ANXIETY vs. REW & DEP  
[menghyp(6),mengp(6),mengzscore(6)] = mengz_JFC(r_a(1+2),r_d(2+2),r12,n);

clc;
mengzscore
mengp


%% Pun-PE Coupling and Learning Bias

idx=1;  
var=squeeze( mean(MEGA_ERP_CORR(:,chani,pun_n2:pun_p3,idx),3)  );  

figure;
subplot(2,2,1); hold on
scatter(TheseData(DEP,GOVAR)-TheseData(DEP,NOGOVAR),var(DEP),'ko'); lsline
[r_a(typei),p_a(typei)]=corr(TheseData(DEP,GOVAR)-TheseData(DEP,NOGOVAR),var(DEP),'rows','complete','type','Spearman');
text(.1,.2,['rho=',num2str(r_a(typei))],'sc');      text(.1,.1,['p=',num2str(p_a(typei))],'sc');

subplot(2,2,3); hold on
scatter(TheseData(DEP,NOGOVAR),var(DEP),'ro'); lsline
[r_a(typei),p_a(typei)]=corr(TheseData(DEP,NOGOVAR),var(DEP),'rows','complete','type','Spearman');
text(.1,.2,['rho=',num2str(r_a(typei))],'sc');      text(.1,.1,['p=',num2str(p_a(typei))],'sc');

subplot(2,2,4); hold on
scatter(TheseData(DEP,GOVAR),var(DEP),'go'); lsline
[r_a(typei),p_a(typei)]=corr(TheseData(DEP,GOVAR),var(DEP),'rows','complete','type','Spearman');
text(.1,.2,['rho=',num2str(r_a(typei))],'sc');      text(.1,.1,['p=',num2str(p_a(typei))],'sc');
 
%%  Topos

for typei=1:4
    if      typei==1,  data=MEGA_ERP;       idx=1;  var=squeeze( MEGA_ERP(:,:,pun_p3,idx)-MEGA_ERP(:,:,pun_n2,idx) ); 
        LIMS=[-15 15];  
    elseif  typei==2,  data=MEGA_ERP_CORR;  idx=1;  var=squeeze( mean(MEGA_ERP_CORR(:,:,pun_n2:pun_p3,idx),3)  );  
        LIMS=[-.1 .1]; 
    elseif  typei==3,  data=MEGA_ERP;       idx=2;  var=squeeze( mean(MEGA_ERP(:,:,rewp1:rewp2,idx),3)  );
        LIMS=[-25 25]; 
    elseif  typei==4,  data=MEGA_ERP_CORR;  idx=2;  var=squeeze( mean(MEGA_ERP_CORR(:,:,rewp1:rewp2,idx),3)  ); 
         LIMS=[-.1 .1];  
    end

   figure; topoplot(squeeze(mean(var,1)),NScan_Chanlocs_60,'maplimits',LIMS); title(num2str(typei))

end


%% TF
rx_dep_anx=repmat(corr( TheseData(DEP,DEPVAR) , TheseData(DEP,ANXVAR)  ,'type','Spearman'),50*751,1);
frex=logspace(.01,1.7,50);

TFchani=1;

TFt1=[200,200];
TFt2=[400,400];
TFf1=[19,8];  % 4, 2
TFf2=[27,14];  % 8, 3

figure; hold on

% TF in CTL group
for paneli=1:2
    
    subplot(2,2,paneli); hold on
    imagesc(tx2disp,[],squeeze( mean(MEGA_POWER{TFchani}(CTL,:,:,paneli),1) ) ); axis xy
    plot([0 0],[1 50],'k:');
    plot([TFt1(paneli), TFt1(paneli)],[TFf1(paneli), TFf2(paneli)],'k:'); 
    plot([TFt2(paneli), TFt2(paneli)],[TFf1(paneli), TFf2(paneli)],'k:');
    plot([TFt1(paneli), TFt2(paneli)],[TFf1(paneli), TFf1(paneli)],'k:'); 
    plot([TFt1(paneli), TFt2(paneli)],[TFf2(paneli), TFf2(paneli)],'k:');
    set(gca,'clim',[-4 4],'xlim',[-500,1000],'ylim',[1 50],'YTick',1:4:length(frex),'YTickLabel',round(frex(1:4:end)));
    
end

% DEP group: corr with symptoms
pval=.05;
for paneli=1:2
    
    subplot(2,2,paneli+2); hold on
    TEMP=squeeze(MEGA_POWER{TFchani}(DEP,:,:,paneli));
    dims=size(TEMP);
    TEMP=reshape(TEMP,dims(1),dims(2)*dims(3));
    
    % Compute corrs with each var
    ToCorr=TheseData(DEP,DEPVAR);
    [rho_dep,p_dep]=corr( TEMP , ToCorr  ,'type','Spearman');
    p_dep(p_dep<=pval)=NaN; p_dep(p_dep>pval)=0; p_dep(isnan(p_dep))=1; 
    rho_dep_2D=reshape(rho_dep,dims(2),dims(3));
    p_dep_2D=reshape(p_dep,dims(2),dims(3));
    corrected_p_dep=Run_Thresh(p_dep_2D);

    ToCorr=TheseData(DEP,ANXVAR);
    [rho_anx,p_anx]=corr( TEMP , ToCorr  ,'type','Spearman');
    p_anx(p_anx<=pval)=NaN; p_anx(p_anx>pval)=0; p_anx(isnan(p_anx))=1; 
    rho_anx_2D=reshape(rho_anx,dims(2),dims(3));    
    p_anx_2D=reshape(p_anx,dims(2),dims(3));
    corrected_p_anx=Run_Thresh(p_anx_2D);
    
    % compute terms
    rsqmean = (rho_dep.*rho_dep + rho_anx.*rho_anx)/2;
    f = (1-rx_dep_anx) ./ (2*(1 - rsqmean));
    f(f>1) = 1;
    h = (1-f.*rsqmean) ./ (1 - rsqmean);
    
    % Fisher transform the two correlations
    z1 = atanh(rho_dep);
    z2 = atanh(rho_anx);
    
    % compute z
    z = (z1 - z2) .* sqrt( (n-3) ./ (2.*(1-rx_dep_anx).*h) );
    
    % JFC EDIT            https://www.mathworks.com/matlabcentral/newsreader/view_thread/298645
    z=abs(z);

    % perform one-tailed test
    p_diff = 1-normcdf(z, 0, 1);
    p_diff_2D=reshape(p_diff,dims(2),dims(3));    
    p_diff(p_diff<=.05)=NaN; p_diff(p_diff>.05)=0; p_diff(isnan(p_diff))=1; p_diff_2D=reshape(p_diff,dims(2),dims(3));
    corrected_p_diff=Run_Thresh(p_diff);
    
    % Display
    if paneli==1
    imagesc(tx2disp,[],  rho_anx_2D  ); axis xy
    contour(tx2disp,1:50,corrected_p_anx,'k','linewidth',2);
    contour(tx2disp,1:50,corrected_p_dep,'r','linewidth',2);
    elseif paneli==2
    imagesc(tx2disp,[],  rho_dep_2D  ); axis xy
    contour(tx2disp,1:50,corrected_p_dep,'k','linewidth',2);    
    contour(tx2disp,1:50,corrected_p_anx,'r','linewidth',2);
    end
    contour(tx2disp,1:50,corrected_p_diff,'m','linewidth',2);

    plot([0 0],[1 50],'k:');
    plot([TFt1(paneli), TFt1(paneli)],[TFf1(paneli), TFf2(paneli)],'k:'); 
    plot([TFt2(paneli), TFt2(paneli)],[TFf1(paneli), TFf2(paneli)],'k:');
    plot([TFt1(paneli), TFt2(paneli)],[TFf1(paneli), TFf1(paneli)],'k:'); 
    plot([TFt1(paneli), TFt2(paneli)],[TFf2(paneli), TFf2(paneli)],'k:');
    set(gca,'clim',[-.5 .5],'xlim',[-500,1000],'ylim',[1 50],'YTick',1:4:length(frex),'YTickLabel',round(frex(1:4:end)));
    clear dims TEMP rho_dep p_dep p_dep_2D corrected_p_dep rsqmean f h z1 z2 z rho_anx p_anx p_anx_2D corrected_p_anx p_diff p_dep p_diff_2D ;
       
end

% Corr with Learning Bias
for paneli=1:2
    TF_ROI(:,paneli)=squeeze( mean(mean( MEGA_POWER{TFchani}(:,TFf1(paneli):TFf2(paneli),...
        find(tx2disp==TFt1(paneli)):find(tx2disp==TFt2(paneli)),paneli)  ,3),2) );
end

GROUP=DEP;
V1=TF_ROI(GROUP,1);    % Main correlate for beh (Punishment Theta)
V1_C=TF_ROI(GROUP,2);  % This one is the contrast (Reward Delta)

figure; hold on

subplot(4,4,[1,2,5,6]); hold on
V2=TheseData(GROUP,GOVAR)-TheseData(GROUP,NOGOVAR);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'k'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[0 10],'ylim',[-1 1]);
title('Go-NoGo')

subplot(4,4,9); hold on
V2=TheseData(GROUP,NOGOVAR);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'r'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[0 10],'ylim',[0 1]);
title('NoGo')
% ---
subplot(4,4,10);
V2=TheseData(GROUP,GOVAR);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'g'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[0 10],'ylim',[0 1]);
title('Go')


%% Mediation     

% https://www.mathworks.com/help/stats/coefficient-standard-errors-and-confidence-intervals.html
% Verifed using SPSS

% This tests for mediation of A-->C as mediated by B
% A = Anxiety
% B = Punishment_Theta
% C = NoGo

Punishment_Theta=TF_ROI(GROUP,1);
Anxiety=TheseData(GROUP,ANXVAR);
NoGoBias=TheseData(GROUP,GOVAR)-TheseData(GROUP,NOGOVAR);

ForSPSS=[Punishment_Theta,Anxiety,NoGoBias];

mdl1 = fitlm(Punishment_Theta,NoGoBias);
[A_coeff] = double(mdl1.Coefficients(2,1:2));
mdl2 = fitlm(Anxiety,NoGoBias);
[B_coeff] = double(mdl2.Coefficients(2,1:2));

% Vars - A & B are unstandardized beta weights towards C, se are the Standard Errors
A=A_coeff(1);
B=B_coeff(1);
seA=A_coeff(2);
seB=B_coeff(2);

% Z-score from the Sobel test > +/-1.96 is significant
sobel = (A*B)/sqrt(((B*B)*(seA*seA))+((A*A)*(seB*seB)))

%% For sLORETA

timewins=[200,250,300,350,400,450,500];

LIMS=quantile(TheseData(DEP,DEPVAR),[.25,.5,.75]);
LOWEST=TheseData(DEP,DEPVAR)<=LIMS(1);
HIGHEST=TheseData(DEP,DEPVAR)>=LIMS(3);
CHANS=[2,19,37,54];
COL={'r','g','b','m'};

% CTL PE Tertile splits and DEP quartile symptom splits
% % figure; 
% % subplot(1,2,1); boxplot(squeeze(MEGA_ERPs_PEs_cts(:,1,:)));
% % subplot(1,2,2); boxplot(squeeze(MEGA_ERPs_PEs_cts(:,2,:)));

chani=find(strcmpi('FCz',{NScan_Chanlocs_60.labels}));
CONDIX=[1,2,3;4,5,6];
figure; hold on
for condi=1:2
    subplot(3,6,CONDIX(condi,:)); hold on
    plot(tx2disp,squeeze( mean(MEGA_ERP_PEs(CTL,chani,:,condi,1),1)  ),'c','linewidth',2);  % low
    plot(tx2disp,squeeze( mean(MEGA_ERP_PEs(CTL,chani,:,condi,2),1)  ),'m','linewidth',2);  % high
    set(gca,'ylim',[-10 15]); plot([0 0],[-10 15],'k:');  plot([-500 1000],[0 0],'k:');
    if condi==1, t1=276; t2=376; else t1=250; t2=350; end
    plot([t1 t1],[10 15],'k:'); plot([t2 t2],[10 15],'k:');
    set(gca,'ylim',[-10 15]); title('CTL')
end

subplot(3,6,7:9); hold on
for ploti=1:4
    chani=CHANS(ploti);
    plot(tx2disp,squeeze( mean(MEGA_ERP(LOWEST,chani,:,2),1)  ),COL{ploti},'linewidth',2);
    set(gca,'ylim',[-10 15]); 
    plot([0 0],[-10 15],'k:');  plot([-500 1000],[0 0],'k:'); title('Lowest Dep')
end
for ploti=1:7
    plot([timewins(ploti) timewins(ploti)],[10 15],'k:')
end

subplot(3,6,10:12); hold on
for ploti=1:4
    chani=CHANS(ploti);
    plot(tx2disp,squeeze( mean(MEGA_ERP(HIGHEST,chani,:,2),1)  ),COL{ploti},'linewidth',2);
    set(gca,'ylim',[-10 15]); 
    plot([0 0],[-10 15],'k:');  plot([-500 1000],[0 0],'k:');title('Highest Dep')
end
for ploti=1:7
    plot([timewins(ploti) timewins(ploti)],[10 15],'k:')
end

for paneli=1:6
    subplot(3,6,12+paneli)
    rewp1_topo=find(tx2disp==timewins(paneli));
    rewp2_topo=find(tx2disp==timewins(paneli+1));
    TempTopo=squeeze( mean(MEGA_ERP(DEP,:,rewp1_topo:rewp2_topo,2),3)  );
    [r,p]=corr(TempTopo,TheseData(DEP,DEPVAR),'rows','complete','type','Spearman');
    p(p>=.05)=NaN;  p(p<.05)=1;   p(isnan(p))=0;
    topoplot(r,NScan_Chanlocs_60,'maplimits',[-.5 .5],'emarker2',{find(p==1),'d','m'});
    title([num2str(timewins(paneli)),' : ',num2str(timewins(paneli+1))]);
end


%% ^^^^^^^^ for sLOR

% % cd('./sLORETA');
% % ANXIETY_RESID=TheseData(DEP,ANXVAR);
% % save( 'ANXIETY_RESID.txt','ANXIETY_RESID', '-ascii');
% % DEPRESSION_RESID=TheseData(DEP,DEPVAR);
% % save( 'DEPRESSION_RESID.txt','DEPRESSION_RESID', '-ascii');
% % NoGoBias=TheseData(DEP,18)-TheseData(DEP,18);
% % save( 'NOGOBIAS.txt','NoGoBias', '-ascii');
% % 
% % XREW=squeeze(MEGA_ERP(DEP,:,rewp1:rewp2,2));
% % for depi=1:size(XREW,1)
% %     REW4sLOR=squeeze(mean(XREW(depi,:,:),3))';
% %     save( [num2str(depi),'_REW4sLOR.txt'],'REW4sLOR', '-ascii');
% %     clear REW4sLOR;
% % end
% % 
% % for paneli=1:6
% %     rewp1_topo=find(tx2disp==timewins(paneli));
% %     rewp2_topo=find(tx2disp==timewins(paneli+1));
% %     TempTopo=squeeze( mean(MEGA_ERP(DEP,:,rewp1_topo:rewp2_topo,2),3)  );
% %     label=[num2str(timewins(paneli)),'_to_',num2str(timewins(paneli+1))];
% %     mkdir(label)
% %     cd(['./',label]);
% %     for depi=1:size(TempTopo,1)
% %         REW4sLOR=squeeze(TempTopo(depi,:))';
% %         save( [num2str(depi),'_REW4sLOR.txt'],'REW4sLOR', '-ascii');
% %         clear REW4sLOR;
% %     end
% %     cd('..');
% % end
% % 
% % XPUN_N2=squeeze(MEGA_ERP(DEP,:,pun_n2,1));
% % for depi=1:size(XPUN_N2,1)
% %     XPUN_N24sLOR=squeeze(XPUN_N2(depi,:,:))';
% %     save( [num2str(depi),'_XPUN_N24sLOR.txt'],'XPUN_N24sLOR', '-ascii');
% %     clear XPUN_N24sLOR;
% % end
% % 
% % XPUN_P3=squeeze(MEGA_ERP(DEP,:,pun_p3,1));
% % for depi=1:size(XPUN_P3,1)
% %     XPUN_P34sLOR=squeeze(XPUN_P3(depi,:,:))';
% %     save( [num2str(depi),'_XPUN_P3.txt'],'XPUN_P34sLOR', '-ascii');
% %     clear XPUN_P34sLOR;
% % end

% % % CTL Pun
% % condi=1;
% % Lo_N2_set=squeeze(MEGA_ERP_PEs(CTL,:,pun_n2,condi,1));
% % Lo_P3_set=squeeze(MEGA_ERP_PEs(CTL,:,pun_p3,condi,1));
% % Hi_N2_set=squeeze(MEGA_ERP_PEs(CTL,:,pun_n2,condi,2));
% % Hi_P3_set=squeeze(MEGA_ERP_PEs(CTL,:,pun_p3,condi,2));
% % for ctli=1:sum(CTL)
% %     Lo_N2=squeeze(Lo_N2_set(ctli,:));
% %     Lo_P3=squeeze(Lo_P3_set(ctli,:));
% %     Hi_N2=squeeze(Hi_N2_set(ctli,:));
% %     Hi_P3=squeeze(Hi_P3_set(ctli,:));
% %     save( [num2str(ctli),'_Lo_N2.txt'],'Lo_N2', '-ascii');
% %     save( [num2str(ctli),'_Lo_P3.txt'],'Lo_P3', '-ascii');
% %     save( [num2str(ctli),'_Hi_N2.txt'],'Hi_N2', '-ascii');
% %     save( [num2str(ctli),'_Hi_P3.txt'],'Hi_P3', '-ascii');
% %     clear Lo_N2 Lo_P3 Hi_N2 Hi_P3;
% % end
% % 
% % % CTL Rew
% % condi=2;
% % Lo_RewP_set=squeeze(MEGA_ERP_PEs(CTL,:,rewp1:rewp2,condi,1));
% % Hi_RewP_set=squeeze(MEGA_ERP_PEs(CTL,:,rewp1:rewp2,condi,2));
% % for ctli=1:sum(CTL)
% %     Lo_RewP=squeeze(Lo_RewP_set(ctli,:));
% %     Hi_RewP=squeeze(Hi_RewP_set(ctli,:));
% %     save( [num2str(ctli),'_Lo_RewP.txt'],'Lo_RewP', '-ascii');
% %     save( [num2str(ctli),'_Hi_RewP.txt'],'Hi_RewP', '-ascii');
% %     clear Lo_RewP Hi_RewP;
% % end

%%  



