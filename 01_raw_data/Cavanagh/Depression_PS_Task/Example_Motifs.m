%%

% Download all data and scripts from this study of depression/anxiety and RL
% That should take care of all script dependencies.
% Run using Matlab 2013b and EEGLab 12_0_2_1b
% Of course, put them in the same folder as this script and ***edit your paths accordingly***

clear all; clc
rootpath='Y:\EEG_Data\PL Cort Depression\';
datapath=[rootpath,'PreProc\'];
savepath='Z:\EXPERIMENTS\Early Career Paper\Motif\';
addpath([rootpath,'FOR UPLOAD\']);  % this is the homedir
modelpath=[rootpath,'FOR UPLOAD\RL Models\']; addpath(modelpath);

[NUM,TXT,RAW]=xlsread([rootpath,'FOR UPLOAD\','Bad_ICAs.xlsx']);  % Hand-identified after help from APPLE

Filz=dir([datapath,'*_PREPROC.mat']);
  
for si=1:length(Filz)
    
    clc; disp(Filz(si).name);
    
    if  1 % ~exist([savepath,num2str(NUM(si,1)),'_PS_ERPs_TF.mat'])
        
        load([datapath,Filz(si).name]);
        
        % Get the good info out of the epochs
        for ai=1:size(EEG.epoch,2)
            for bi=1:size(EEG.epoch(ai).eventlatency,2)
                if EEG.epoch(ai).eventlatency{bi}==0
                    EEG.epoch(ai).Trial=EEG.epoch(ai).eventID_MATRIX{bi}(1);
                    EEG.epoch(ai).Stim=EEG.epoch(ai).eventID_MATRIX{bi}(2);
                    EEG.epoch(ai).Hand=EEG.epoch(ai).eventID_MATRIX{bi}(3);
                    EEG.epoch(ai).Acc=EEG.epoch(ai).eventID_MATRIX{bi}(4);
                    EEG.epoch(ai).RT=EEG.epoch(ai).eventID_MATRIX{bi}(5);
                    EEG.epoch(ai).FB=EEG.epoch(ai).eventID_MATRIX{bi}(6);
                end
            end
        end
        
        %% Model data from the ID_MATRIX (Even tho some trials have been now thrown out of the EEG: That's OK - it'll merge intelligently).
        
        MODEL_RL;
        
        %% Now manage the EEG
        
        % Remove bad ICA
        bad_ICAs_To_Remove=NUM(si,2:end);
        EEG = pop_subcomp( EEG,  bad_ICAs_To_Remove(~isnan(bad_ICAs_To_Remove)), 0);
        
        % Set up times for each
        tx=-2000:1000/EEG.srate:1998;
        B1=find(tx==-300);  B2=find(tx==-200);
        T1=find(tx==-500);  T2=find(tx==1000);
        tx2disp=-500:2:1000;
        
        % ---------- % ---------- % ----------
        % ---------- ERP stuff
        % ---------- % ---------- % ----------
        
        % Spatial filtering makes things pretty interesting too...
% %         X = [EEG.chanlocs.X]; Y = [EEG.chanlocs.Y]; Z = [EEG.chanlocs.Z];
% %         [EEG.data,G,H] = laplacian_perrinX(EEG.data,X,Y,Z,[],1e-6);

        
        dims=size(EEG.data);
        
        for frexi=1:6
            
            if     frexi==1, FILT=eegfilt(EEG.data,500,[],4);  FILT=eegfiltfft(FILT,500,1,[]);
            elseif frexi==2, FILT=eegfilt(EEG.data,500,[],8);  FILT=eegfilt(FILT,500,4,[]);
            elseif frexi==3, FILT=eegfilt(EEG.data,500,[],12); FILT=eegfilt(FILT,500,8,[]);
            elseif frexi==4, FILT=eegfilt(EEG.data,500,[],30); FILT=eegfilt(FILT,500,12,[]);
            elseif frexi==5, FILT=eegfilt(EEG.data,500,[],50); FILT=eegfilt(FILT,500,30,[]);
            elseif frexi==6, FILT=eegfilt(EEG.data,500,[],50); FILT=eegfiltfft(FILT,500,.1,[]);
            end
            
            % Just for fun - - Hilbertize! - - being overly cautious using the loop to ensure this happens over the TIME dimension.
            for ci=1:dims(1)
                FILT_HILBERT(ci,:)=abs(hilbert( squeeze(FILT(ci,:)) )).^2;
            end
            
            FILT=reshape(FILT,dims(1),dims(2),dims(3));
            FILT_HILBERT=reshape(FILT_HILBERT,dims(1),dims(2),dims(3));
            
            % Basecor your ERPs here so they are pretty.
            BASE=squeeze(  mean(FILT(:,B1:B2,:),2)  );
            BASE_FILT_HILBERT=squeeze(  mean(FILT_HILBERT(:,B1:B2,:),2)  );
            for ai=1:dims(1)
                FILT(ai,:,:)=squeeze(FILT(ai,:,:))-repmat( BASE(ai,:),dims(2),1 );
                FILT_HILBERT(ai,:,:)=squeeze(FILT_HILBERT(ai,:,:))-repmat( BASE_FILT_HILBERT(ai,:),dims(2),1 );
            end
            
            % Corr with PE
            for chani=1:size(FILT,1)
                for pei=1:2
                    pe4corr=ID_MATRIX_V2(ID_MATRIX_V2(:,6)==pei-1,7);
                    erp4corr=squeeze(FILT(chani,T1:T2,ID_MATRIX_V2(:,6)==pei-1));
                    ERP_CORR(frexi,chani,:,pei) = corr(erp4corr',pe4corr,'type','Spearman','rows','complete');
                    clear erp4corr;
                    erp4corr=squeeze(FILT_HILBERT(chani,T1:T2,ID_MATRIX_V2(:,6)==pei-1));
                    ERP_CORR_FILT_HILBERT(frexi,chani,:,pei) = corr(erp4corr',pe4corr,'type','Spearman','rows','complete');
                    clear pe4corr erp4corr;
                end
            end
            
            % Get ERPs
            ERPs(frexi,:,:,1)=squeeze(mean(  FILT(:,T1:T2,ID_MATRIX_V2(:,6)==0) ,3));
            ERPs(frexi,:,:,2)=squeeze(mean(  FILT(:,T1:T2,ID_MATRIX_V2(:,6)==1) ,3));
            ERPs_FILT_HILBERT(frexi,:,:,1)=squeeze(mean(  FILT_HILBERT(:,T1:T2,ID_MATRIX_V2(:,6)==0) ,3));
            ERPs_FILT_HILBERT(frexi,:,:,2)=squeeze(mean(  FILT_HILBERT(:,T1:T2,ID_MATRIX_V2(:,6)==1) ,3));
            
            % Split by hi and lo PEs
            ID_MATRIX_V2(:,8)=1:length(ID_MATRIX_V2);
            for pei=1:2
                pe4corr=[ID_MATRIX_V2(ID_MATRIX_V2(:,6)==pei-1,8) , abs(ID_MATRIX_V2(ID_MATRIX_V2(:,6)==pei-1,7)) ];
                TERTILES=quantile(pe4corr(:,2),[.33,.66]);
                Low_Idx=pe4corr(pe4corr(:,2)<=TERTILES(1),1);
                Hi_Idx=pe4corr(pe4corr(:,2)>=TERTILES(2),1);
                ERPs_PEs_cts(pei,:)=[length(Low_Idx),length(Hi_Idx)];
                ERPs_PEs(frexi,:,:,pei,1)=squeeze(mean(  FILT(:,T1:T2,Low_Idx) ,3));
                ERPs_PEs(frexi,:,:,pei,2)=squeeze(mean(  FILT(:,T1:T2,Hi_Idx) ,3));
                ERPs_PEs_FILT_HILBERT(frexi,:,:,pei,1)=squeeze(mean(  FILT_HILBERT(:,T1:T2,Low_Idx) ,3));
                ERPs_PEs_FILT_HILBERT(frexi,:,:,pei,2)=squeeze(mean(  FILT_HILBERT(:,T1:T2,Hi_Idx) ,3));
                clear pe4corr TERTILES Low_Idx Hi_Idx;
            end
            
            clear FILT FILT_HILBERT;
            
        end
        
        
        save([savepath,num2str(NUM(si,1)),'_PEmotifs.mat'],'ERPs','ERPs_PEs','ERP_CORR','ERPs_PEs_cts',...
                                                           'ERPs_FILT_HILBERT','ERP_CORR_FILT_HILBERT','ERPs_PEs_FILT_HILBERT');
        
        clearvars -except si *path TIME NUM TXT RAW Filz;
        
    end
end


%%
clear all; clc
rootpath='Z:\EXPERIMENTS\Early Career Paper\Motif\';  addpath(rootpath);  cd(rootpath);
load('Y:\EEG_Data\PL Cort Depression\FOR UPLOAD\NScan_Chanlocs_60.mat','NScan_Chanlocs_60');
Filz=dir([rootpath,'*_PEmotifs.mat']);  

for si=1:length(Filz)  
    
    clc; disp(Filz(si).name);
    load([rootpath,Filz(si).name]);
    
    % Invert -PE to abs(-PE)
    ERP_CORR(:,:,:,1)=-1.*ERP_CORR(:,:,:,1);
    ERP_CORR_FILT_HILBERT(:,:,:,1)=-1.*ERP_CORR_FILT_HILBERT(:,:,:,1);
   
    MEGA_ERP(si,:,:,:,:)=ERPs;
    MEGA_ERP_CORR(si,:,:,:,:)=ERP_CORR;
    MEGA_ERP_PEs(si,:,:,:,:,:)=ERPs_PEs;
    
    MEGA_ERP_FH(si,:,:,:,:)=ERPs_FILT_HILBERT;
    MEGA_ERP_CORR_FH(si,:,:,:,:)=ERP_CORR_FILT_HILBERT;
    MEGA_ERP_PEs_FH(si,:,:,:,:,:)=ERPs_PEs_FILT_HILBERT;

    clear ERPs ERP_CORR ERPs_PEs ERPs_FILT_HILBERT ERP_CORR_FILT_HILBERT ERPs_PEs_FILT_HILBERT;
end

size(MEGA_ERP)   % incor, cor
size(MEGA_ERP_CORR)
size(MEGA_ERP_PEs)


%% 

tx2disp=-500:2:1000;
rewp1=find(tx2disp==250);
rewp2=find(tx2disp==350);

pun_n1=find(tx2disp==100);
pun_p2=find(tx2disp==220);
pun_n2=find(tx2disp==276);
pun_p3=find(tx2disp==376);

figure; topoplot(squeeze(mean(mean(MEGA_ERP(:,6,:,pun_n1,1),4),1)),NScan_Chanlocs_60,'maplimits',[-7 -3]); title('N1')

PEcol={'k:','k'};
PEcol2={'r:','r' ; 'b:','b' ;'c:','c' ;'g:','g' ;'m:','m'; 'k:','k';'g:','g' ;'m:','m'};
MULTI=[1,1,5,10,40,1,1,1];

chani=find(strcmpi('FCz',{NScan_Chanlocs_60.labels}));

figure;
for PEi=1:2
    subplot(2,1,1);  hold on
    plot(tx2disp,squeeze(mean(MEGA_ERP_PEs(:,6,chani,:,1,PEi),1)),PEcol{PEi},'linewidth',2);
end
for PEi=1:2
    subplot(2,1,2);  hold on
    for filti=1:5
        
        if filti==5
        datx=squeeze(mean(MEGA_ERP_PEs(:,filti,chani,find(tx2disp==0):find(tx2disp==250),1,PEi),1)).*MULTI(filti);
        plot(tx2disp,[NaN(1,251),datx',NaN(1,751-251-126)],PEcol2{filti,PEi},'linewidth',2);             
        else
        plot(tx2disp,squeeze(mean(MEGA_ERP_PEs(:,filti,chani,:,1,PEi),1)).*MULTI(filti),PEcol2{filti,PEi},'linewidth',2);
        end
         
    end
end




%%
