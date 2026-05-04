%%
clear all; clc
rootpath='Y:\EEG_Data\PL Cort Depression\';
savepath=[rootpath,'PreProc\']; cd(savepath);
addpath([rootpath,'FOR UPLOAD\']);  % this is the homedir
datapath=[rootpath,'FOR UPLOAD\Raw Data\'];
modelpath=[rootpath,'FOR UPLOAD\RL Models\'];
addpath(genpath('Y:\Programs\eeglab12_0_2_1b'));
locpath=('Y:\Programs\eeglab12_0_2_1b\plugins\dipfit2.2\standard_BESA\standard-10-5-cap385.elp');

Filz=dir([datapath,'*.mat']);

for si=1:length(Filz)
    
    % ---------------- TRIGGERS
    AB={'10','11','12','13'}; % AB, AB, BA, BA
    CD={'14','15','16','17'};
    EF={'18','19','20','21'};
    TRN_Left_Better=[10,11,14,15,18,19];
    TRN_Right_Better=[12,13,16,17,20,21];
    FB_Types={'94','104'};   % Cor, Incor
    % ----------------
    for tsti=1:30
        TST{tsti}=num2str(200+tsti-1);
    end
    All_STIM={AB{:},CD{:},EF{:},TST{:}};
    TST_Left_Better=[200:2:228];
    TST_Right_Better=[201:2:229];
    % ----------------
    Left_Better=[TRN_Left_Better,TST_Left_Better];
    Right_Better=[TRN_Right_Better,TST_Right_Better];
    % ----------------
    
    filename=Filz(si).name;
    subno=str2num(filename(1:3));
    
    if ~exist([savepath,num2str(subno),'_PREPROC.mat']);
        
        % Load
        load([datapath,filename]);
        
        % Get the good info out of the events
        Trial=0;  ID_MATRIX=[];
        for ai=1:size(EEG.event,2)
            FB=NaN; Hand=NaN; RT_time=NaN; Stim=NaN; Stim_time=NaN; Acc=NaN;
            if any(strcmp(EEG.event(ai).type,FB_Types))
                FB=str2num(EEG.event(ai).type);
                Trial=Trial+1;
                % Get prior response
                if length(EEG.event(ai-1).type)==7
                    if strcmp(EEG.event(ai-1).type(1:6),'keypad')
                        Hand=str2num(EEG.event(ai-1).type(7));
                        RT_time=EEG.event(ai-1).latency;
                        % Now for stim pair.  Since they may have pressed a few buttons, try this loop:
                        stimbreak=0;
                        for stimi=1:4
                            if stimbreak==0;
                                if any(strcmp(EEG.event(ai-1-stimi).type,All_STIM)) ;
                                    Stim=str2num(EEG.event(ai-1-stimi).type);
                                    Stim_time=EEG.event(ai-1-stimi).latency;
                                    stimbreak=1;
                                end
                            end
                        end
                        % Code to make more sense
                        if any(Stim==Left_Better)
                            if Hand==1,
                                Acc=0;
                            elseif Hand==2,
                                Acc=1;
                            end
                        elseif any(Stim==Right_Better)
                            if Hand==1,
                                Acc=1;
                            elseif Hand==2,
                                Acc=0;
                            end
                        end
                    end
                end
                % Now put this together
                ID_MATRIX=[ID_MATRIX;Trial,Stim,Hand,Acc,RT_time-Stim_time,FB];
                EEG.event(ai).ID_MATRIX=[Trial,Stim,Hand,Acc,RT_time-Stim_time,FB];
            end
        end

        % Channels
        EEG = pop_chanedit(EEG,  'lookup', locpath);
        EEG = eeg_checkset( EEG );
        
        
        % Epoch
        EEG = pop_epoch( EEG, FB_Types, [-2  2], 'newname', 'Epochs', 'epochinfo', 'yes');
        EEG = eeg_checkset( EEG );
        
        % Save VEOG off to the side   - - whoops, dammit, most of these were mislabeled as HEOG during datacollection. 
        EEG.VEOG=squeeze(EEG.data(find(strcmpi('HEOG',{EEG.chanlocs.labels})),:,:));

        % Strip bads: 60=CB1, 64=CB2, 65=HEOG, 66=VEOG  |||  33=M1, 43=M2, 
        EEG = pop_select(EEG,'nochannel',[find(strcmpi('CB1',{EEG.chanlocs.labels})) find(strcmpi('CB2',{EEG.chanlocs.labels})) ...
            find(strcmpi('HEOG',{EEG.chanlocs.labels})) find(strcmpi('EKG',{EEG.chanlocs.labels})) find(strcmpi('VEOG',{EEG.chanlocs.labels}))]);
        
        % Remove mean
        EEG = pop_rmbase(EEG,[],[]);
        
        
        % ----------------------
        % Setup APPLE to interp chans, reject epochs, & ID bad ICs.  Output will be LM ref'd and ICA'd.
        TASK='PS';
        eeg_chans=1:62;   % includes mastoids, which APPLE takes care of...
        Do_ICA=1;
        ref_chan=19;  % Re-Ref to FCz   [WEIRD STEP, BUT THIS IS FOR FASTER, which is a part of APPLE]
        EEG = pop_reref(EEG,ref_chan,'keepref','on'); % ------ don't actually re-ref.  This will keep it LM the whole time.
        
        % Run APPLE
        [EEG,EEG.bad_chans,EEG.bad_epochs,EEG.bad_ICAs,EEG.PVs]=APPLE_Dep(EEG,eeg_chans,ref_chan,Do_ICA,subno,EEG.VEOG,TASK);
        
        % Save ------ data are now 60 channels referenced to LM
        save([savepath,num2str(subno),'_PREPROC.mat'],'EEG','ID_MATRIX');
        
        clearvars -except Filz *path FB si;
        % ----------------------


    end
end

%%

STEP2_PROCESS


%%

