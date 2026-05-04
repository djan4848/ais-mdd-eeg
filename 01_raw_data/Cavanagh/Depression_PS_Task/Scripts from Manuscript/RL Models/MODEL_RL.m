%% Model RL Mem

% Fitting models after APPLE artifact rejection, so it's easier to edit models without running pre-proc again.

% Here's what we did in Step 1:
% % 1=Trial
% % 2=Stim
% % 3=Hand
% % 4=Acc
% % 5=RT
% % 6=FB
    
% Now get just the goods
for ai=1:size(ID_MATRIX,1)
    BEH4MODEL(ai,1)=ID_MATRIX(ai,1); % Trial
    BEH4MODEL(ai,2)=ID_MATRIX(ai,2); % Stim
    BEH4MODEL(ai,3)=ID_MATRIX(ai,4); % Response Accuracy
    BEH4MODEL(ai,4)=ID_MATRIX(ai,6); % FB Recieved
end
% Limit to only TRN data
BEH4MODEL=BEH4MODEL(BEH4MODEL(:,2)>0,:);
% Kill NaNs
BEH4MODEL=BEH4MODEL(~isnan(BEH4MODEL(:,3)),:);
BEH4MODEL=BEH4MODEL(~isnan(BEH4MODEL(:,4)),:);
% Re-code stim sets to simpler version
BEH4MODEL(BEH4MODEL(:,2)<14,2)=1; % AB
BEH4MODEL(BEH4MODEL(:,2)>17,2)=3; % EF 
BEH4MODEL(BEH4MODEL(:,2)>3,2)=2; % CD 
% Re-code FB to simpler version
BEH4MODEL(BEH4MODEL(:,4)==104,4)=0; % Pun
BEH4MODEL(BEH4MODEL(:,4)==94,4)=1; % Rew

% ******** models for comparison
% Foil - (inline) No model, just data with random selections
% Vanilla - l-rate & softmax model
% Vanilla2 - 2 parameter l-rate & softmax model  

for modeli=1:3
    
    if     modeli==1,   Modeltype='Foil';
    elseif modeli==2,   Modeltype='Vanilla';
    elseif modeli==3,   Modeltype='Vanilla2';
    end
    
    goodruns=0;
    while goodruns==0;
        [xfinal,ffinal,exitflag,xstart]  = SETUP_RLMEM(BEH4MODEL,Modeltype);
        if sum(~isnan(ffinal))~=0
            startidx=find(ffinal==min(ffinal));
            params_out=xfinal(min(startidx),:);
            LLE_out=ffinal(min(startidx),:);
            goodruns=1;
        end
    end
    % Recover info
    [PE,Q,Params]=SETUP_RLMEM_Recover(BEH4MODEL,Modeltype,params_out);
    
    MODEL{modeli}.PE=PE;
    MODEL{modeli}.Q=Q;
    MODEL{modeli}.Params=Params;
    MODEL{modeli}.LLE=LLE_out;
    
    clear goodruns startidx params_out LLE_out xfinal ffinal exitflag xstart PE Params LLE_out;
    
end

% Merge model with EEG
modeli=3; 

BEH4MODEL(:,5)=MODEL{modeli}.PE;
for ai=1:size(EEG.epoch,2)
    EEG.epoch(ai).PE=[];  % initialize as empty first
    if any(EEG.epoch(ai).Trial==BEH4MODEL(:,1))  % TRN only
        idx=find(BEH4MODEL(:,1)==EEG.epoch(ai).Trial);
        EEG.epoch(ai).PE=BEH4MODEL(idx,5); clear idx;
    end
    if isempty(EEG.epoch(ai).PE) , EEG.epoch(ai).PE=NaN; end
end

clear BEH4MODEL ID_MATRIX;

% Make new ID_MATRIX_V2
for ai=1:size(EEG.epoch,2)
    ID_MATRIX_V2(ai,1)=EEG.epoch(ai).Trial;
    ID_MATRIX_V2(ai,2)=EEG.epoch(ai).Stim;
    ID_MATRIX_V2(ai,3)=EEG.epoch(ai).Hand;
    ID_MATRIX_V2(ai,4)=EEG.epoch(ai).Acc;
    ID_MATRIX_V2(ai,5)=EEG.epoch(ai).RT;
    ID_MATRIX_V2(ai,6)=EEG.epoch(ai).FB;
    ID_MATRIX_V2(ai,7)=EEG.epoch(ai).PE;
end
% Re-code feedback to make it easier in Step 2
ID_MATRIX_V2(ID_MATRIX_V2(:,6)==94,6)=1;
ID_MATRIX_V2(ID_MATRIX_V2(:,6)==104,6)=0;
% clear NaN trials
for ai=1:size(EEG.epoch,2)
    if isnan(ID_MATRIX_V2(ai,4)), ID_MATRIX_V2(ai,2:end)=NaN; end
end
ID_MATRIX_V2_HDR={'Trial','Stim','Hand','Acc','RT','FB','PE'};

%%


