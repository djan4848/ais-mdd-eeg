%%
clear all; clc
rootpath='Y:\EEG_Data\PL Cort Depression\';
datapath=[rootpath,'PreProc\'];
savepath=[rootpath,'Processed Data\'];
addpath([rootpath,'FOR UPLOAD\']);  % this is the homedir
modelpath=[rootpath,'FOR UPLOAD\RL Models\']; addpath(modelpath);

[NUM,TXT,RAW]=xlsread([rootpath,'FOR UPLOAD\','Bad_ICAs.xlsx']);  % Hand-identified after help from APPLE

Filz=dir([datapath,'*_PREPROC.mat']);

for si=length(Filz):-1:1
    
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
        
        % Setup Wavelet Params
        num_freqs=50;
        frex=logspace(.01,1.7,num_freqs);
        s=logspace(log10(3),log10(10),num_freqs)./(2*pi*frex);
        t=-2:1/EEG.srate:2;
        
        % Definte Convolution Parameters
        dims = size(EEG.data);
        n_wavelet = length(t);
        n_data = dims(2)*dims(3);
        n_convolution = n_wavelet+n_data-1;
        n_conv_pow2 = pow2(nextpow2(n_convolution));
        half_of_wavelet_size = (n_wavelet-1)/2;
        
        % Pick channel
        chani=[19,28];    % 19=FCz 28=Cz
        
        % Get FFT of data
        for cidx=1:2
            EEG_fft = fft(reshape(EEG.data(chani(cidx),:,:),1,n_data),n_conv_pow2);
            
            for fi=1:num_freqs
                
                wavelet = fft( exp(2*1i*pi*frex(fi).*t) .* exp(-t.^2./(2*(s(fi)^2))) , n_conv_pow2 );  % sqrt(1/(s(fi)*sqrt(pi))) *
                
                % convolution
                EEG_conv = ifft(wavelet.*EEG_fft);
                EEG_conv = EEG_conv(1:n_convolution);
                EEG_conv = EEG_conv(half_of_wavelet_size+1:end-half_of_wavelet_size);
                EEG_conv = reshape(EEG_conv,dims(2),dims(3));

                % Corr with PE
                for pei=1:2
                    pe4corr=ID_MATRIX_V2(ID_MATRIX_V2(:,6)==pei-1,7);
                    pwr4corr=abs(EEG_conv(T1:T2,ID_MATRIX_V2(:,6)==pei-1)).^2;
                    PWR_CORR{cidx}(fi,:,pei) = corr(pwr4corr',pe4corr,'type','Spearman','rows','complete');
                    clear pe4corr pwr4corr;
                end
                
                % Get power by condi (different time windows)
                temp_POWER1 = mean(abs(EEG_conv(T1:T2,ID_MATRIX_V2(:,6)==0)).^2,2);
                temp_POWER2 = mean(abs(EEG_conv(T1:T2,ID_MATRIX_V2(:,6)==1)).^2,2);
                
                % Get baseline by condi (different time windows)
                temp_BASE = mean(mean(abs(EEG_conv(B1:B2,:)).^2,1),2);
                
                % dB correct power by base (different time windows)
                POWER{cidx}(fi,:,1) = mean(10*(log10(temp_POWER1) - log10(repmat(temp_BASE,size(tx2disp,2),1))),2);
                POWER{cidx}(fi,:,2) = mean(10*(log10(temp_POWER2) - log10(repmat(temp_BASE,size(tx2disp,2),1))),2);
                
                % Get ITPC by condi (different time windows)
                ITPC{cidx}(fi,:,1) = abs(mean(exp(1i*(  angle(EEG_conv(T1:T2,ID_MATRIX_V2(:,6)==0))  )),2));
                ITPC{cidx}(fi,:,2) = abs(mean(exp(1i*(  angle(EEG_conv(T1:T2,ID_MATRIX_V2(:,6)==0))  )),2));
                
                % ----------------
                clear EEG_conv wavelet temp*;
                
            end
            clear EEG_fft;
        end
        
        % ---------- % ---------- % ----------
        % ---------- ERP stuff
        % ---------- % ---------- % ----------
        
        dims=size(EEG.data);
        EEG.data=eegfilt(EEG.data,500,[],20);
        EEG.data=eegfiltfft(EEG.data,500,.1,[]);
        EEG.data=reshape(EEG.data,dims(1),dims(2),dims(3));
        
        % Basecor your ERPs here so they are pretty.
        BASE=squeeze(  mean(EEG.data(:,B1:B2,:),2)  );
        for ai=1:dims(1)
            EEG.data(ai,:,:)=squeeze(EEG.data(ai,:,:))-repmat( BASE(ai,:),dims(2),1 );
        end
        
        % Corr with PE
        for chani=1:size(EEG.data,1)
            for pei=1:2
                pe4corr=ID_MATRIX_V2(ID_MATRIX_V2(:,6)==pei-1,7);
                erp4corr=squeeze(EEG.data(chani,T1:T2,ID_MATRIX_V2(:,6)==pei-1));
                ERP_CORR(chani,:,pei) = corr(erp4corr',pe4corr,'type','Spearman','rows','complete');
                clear pe4corr erp4corr;
            end
        end

        % Get ERPs
        ERPs(:,:,1)=squeeze(mean(  EEG.data(:,T1:T2,ID_MATRIX_V2(:,6)==0) ,3));
        ERPs(:,:,2)=squeeze(mean(  EEG.data(:,T1:T2,ID_MATRIX_V2(:,6)==1) ,3));
                
        % Split by hi and lo PEs
        ID_MATRIX_V2(:,8)=1:length(ID_MATRIX_V2);
        for pei=1:2
            pe4corr=[ID_MATRIX_V2(ID_MATRIX_V2(:,6)==pei-1,8) , abs(ID_MATRIX_V2(ID_MATRIX_V2(:,6)==pei-1,7)) ];
            TERTILES=quantile(pe4corr(:,2),[.33,.66]);
            Low_Idx=pe4corr(pe4corr(:,2)<=TERTILES(1),1);
            Hi_Idx=pe4corr(pe4corr(:,2)>=TERTILES(2),1);
            ERPs_PEs_cts(pei,:)=[length(Low_Idx),length(Hi_Idx)];
            ERPs_PEs(:,:,pei,1)=squeeze(mean(  EEG.data(:,T1:T2,Low_Idx) ,3));  
            ERPs_PEs(:,:,pei,2)=squeeze(mean(  EEG.data(:,T1:T2,Hi_Idx) ,3)); 
            clear pe4corr TERTILES Low_Idx Hi_Idx;
        end

        
        save([savepath,num2str(NUM(si,1)),'_PS_ERPs_TF.mat'],'ERPs','POWER','ITPC','PWR_CORR','ERP_CORR','ID_MATRIX_V2','MODEL','ERPs_PEs_cts','ERPs_PEs');
        
        clearvars -except si *path TIME NUM TXT RAW Filz;
        
    end
end



%%
