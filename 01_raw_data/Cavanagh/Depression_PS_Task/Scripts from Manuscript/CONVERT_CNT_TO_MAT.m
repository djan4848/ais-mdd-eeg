%%
clear all; clc
rootpath='Y:\EEG_Data\PL Cort Depression\';  addpath(rootpath);
datapath=[rootpath,'EEG Data\'];
savepath='Y:\EEG_Data\PL Cort Depression\FOR UPLOAD\Raw Data\';   cd(savepath);

Filz=dir([datapath,'T1',' Ready\','*.cnt']);

for si=1:length(Filz)
    
    filename=Filz(si).name;
    subno=str2num(filename(1:3));
    
    % Load
    EEG = pop_loadcnt([datapath,'T1',' Ready\',filename], 'dataformat', 'int32', 'keystroke', 'on');
    EEG.comments = [];
    EEG = eeg_checkset( EEG );
    
    % Save
    save([savepath,num2str(subno),'.mat'],'EEG');
    
    % Housekeeping
    clear EEG filename subno;
    
end