function [EEG,bad_chans,bad_epochs,bad_ICAs,PV]=APPLE_Dep(EEG,eeg_chans,ref_chan,Do_ICA,subno,VEOG,TASK)
% Algorithmic Pre-Processing Line for EEG
% Intellectual Property of James F Cavanagh   jcavanagh@unm.edu    2013
% Use eeglab12_0_2_1b and in the plugins folder include the following:
% FASTER 1.2.3:        http://sourceforge.net/projects/faster/ 
% ADJUST:     http://www.unicog.org/pm/pmwiki.php/MEG/RemovingArtifactsWithADJUST
% Gratton Eyeblink Correction:      can't find the website, but it doesn't work well and I'm going to remove it anyways.  Just have 1 for Do_ICA always.

% ===================
%   MANDATORY INPUT  
% ===================
% EEG        - The eponymous EEGLab array
% eeg_chans  - vector of EEG channels (exclude VEOG, HEOG, anything else here)
% ref_chan   - Call_APPLE should have re-ref'd the data to Fz or FCz     
% Do_ICA     - Do ICAs or not?  If no, it will run Gratton regression if there is VEOG
%             
% ====================
% OPTIONAL PARAMETERS
% ====================
% SubjID     - Subject ID for saving the output jpeg.  Optional - will be set to 0 if empty
% VEOG       - The vector of VEOG stripped from the EEG.data structure for
%               use in ID'ing ICA blinks.
%
% ==============
%     OUTPUT    
% ==============
% EEG - interpolated with bad epochs rejected.  If Gratton, eyeblinks removed.
% bad_chans       
% bad_epochs     
% bad_ICAs  
 
% Start clock
tic

% Get stuff
SubjID=num2str(subno); 

% Get dimensions of EEG data matrix
dims=size(EEG.data);

% Get Vertex Site
for ai=1:dims(1), Z(ai)=EEG.chanlocs(ai).Z;  end
Vertex=find(Z==max(Z));  clear Z;
% Get ERP % Topo of these data prior to fixen's
TEMPPRE = pop_reref( EEG, []);
PreFixERP=eegfilt(squeeze(mean(TEMPPRE.data(Vertex,:,:),3)),TEMPPRE.srate,[],20);
PreFixERP=PreFixERP-repmat(mean(PreFixERP),1,length(PreFixERP));
% Get times irrespective of sample rate
T1=find( abs(TEMPPRE.times-300) == min(abs(TEMPPRE.times-300)) )  ;
T2=find( abs(TEMPPRE.times-400) == min(abs(TEMPPRE.times-400)) )  ;
PreFixTopo=squeeze(mean(mean(TEMPPRE.data(:,T1:T2,:),2),3));  % Topo w/ blinks
clear TEMPPRE; 

%% ID bad channels

% EEGLab Function
tempeeg=EEG;  % save the real data as an archive
[EEG, indelec, measure] = pop_rejchan( EEG, 'elec', eeg_chans); % process on the 'EEG' set
clear EEG; EEG=tempeeg; clear tempeeg;  % save what was done to the 'EEG' set, then erase it and replace with archive

% FASTER
chan = channel_properties(EEG, eeg_chans, ref_chan);
chan_exceeded_threshold = min_z_JFC(chan);  % Cols are: 1) weighted correlation, weighted variance, Hurst
FASTER_bad_chans = find(logical(chan_exceeded_threshold(:,2)+chan_exceeded_threshold(:,3)));

% Combine unique elements
TOTAL_bad_chans=unique([FASTER_bad_chans(:);indelec(:)]);

% Hard-code some bad ones
if subno==543, FASTER_bad_chans=unique([FASTER_bad_chans(:);find(strcmpi('FC1',{EEG.chanlocs.labels}))]); end
if subno==548, FASTER_bad_chans=unique([FASTER_bad_chans(:);find(strcmpi('FC1',{EEG.chanlocs.labels}))]); end
if subno==558, FASTER_bad_chans=unique([FASTER_bad_chans(:);find(strcmpi('FC1',{EEG.chanlocs.labels}))]); end
if subno==561, FASTER_bad_chans=unique([FASTER_bad_chans(:);find(strcmpi('F2',{EEG.chanlocs.labels}))]); end
if subno==562, FASTER_bad_chans=unique([FASTER_bad_chans(:);find(strcmpi('FC1',{EEG.chanlocs.labels}));find(strcmpi('F2',{EEG.chanlocs.labels}))]); end

% INTERPOLATE
if ~isempty(TOTAL_bad_chans)
    EEG.data=double(EEG.data);
    EEG = pop_interp(EEG,FASTER_bad_chans,'spherical');        % NOTE HERE ONLY USING FASTER AS indelec GETS ODD ON SOME!!!
end

bad_chans{1}=FASTER_bad_chans;
bad_chans{2}=indelec;
bad_chans{3}=TOTAL_bad_chans;

%% NOW re-ref to LM - after interpolation and before rejection (pop_autorej requires it)
EEG = pop_reref(EEG,[find(strcmpi('M1',{EEG.chanlocs.labels})) find(strcmpi('M2',{EEG.chanlocs.labels}))]);
eeg_chans=1:max(eeg_chans)-2;
dims=size(EEG.data);

%% ID bad epochs 

% % % EEGLab Function
% % tempeeg=EEG; % same as above - this takes a while though
% % [EEG, rmepochs] = pop_autorej(EEG,'nogui','on');
% % % - here
% % clear EEG; EEG=tempeeg; clear tempeeg;
% % autorej_bad_epochs=zeros(EEG.trials,1);  % vb Vectorize the output
% % autorej_bad_epochs(sort(rmepochs))=1;

% FASTER
epoch = epoch_properties(EEG,eeg_chans);
epoch_exceeded_threshold = min_z_JFC(epoch);  % Cols are: 1) mean epoch deviation, 2) epoch variance, 3) max amplitude
FASTER_bad_epochs = logical(epoch_exceeded_threshold(:,1)+epoch_exceeded_threshold(:,2)+epoch_exceeded_threshold(:,3)); % ANYTHING marked as bad is bad

% Combine unique elements
TOTAL_bad_epochs=logical(FASTER_bad_epochs);

% REJECT
binarized=zeros(1,EEG.trials);
binarized(FASTER_bad_epochs)=1;    % Only the FASTER ones
EEG = pop_rejepoch(EEG,binarized,0);
goodepochs=logical(1-binarized);

EP2REJ=1;
bad_epochs{1}=FASTER_bad_epochs; 
bad_epochs{2}=99;
bad_epochs{3}=TOTAL_bad_epochs;

%%  Deal with blinks

if Do_ICA==1
    
    % Calculate kC^2 = # of data points needed
    k=25; % Suggested by Onton et al.
    C=dims(1)-length(TOTAL_bad_chans);  % n good independent channels
    sizeneeded=C^2*k;
    epochsneeded=round(sizeneeded/EEG.srate);  % # of epochs needed for a stable ICA solution

    % ##### ##### ICA ##### #####
    EEG = pop_runica(EEG,'icatype','runica','pca',size(EEG.data,1)-length(FASTER_bad_chans));     
    
    % ADJUST
% %     EEG.icaact = eeg_getica(EEG);
% %     [art, horiz, vert, blink, disc, soglia_DV, diff_var, soglia_K,...
% %         meanK, soglia_SED, SED, soglia_SAD, SAD, soglia_GDSF, GDSF, soglia_V, nuovaV]=ADJUST(EEG,'junkfile');
    bad_ADJUST_ICAs=99;
    
    EEG.icaact = (EEG.icaweights*EEG.icasphere)*EEG.data(EEG.icachansind,:);

    % Do VEOG correlation
    if ~isempty(VEOG)
        for ai=1:size(EEG.icaact,1)
            temp=squeeze(EEG.icaact(ai,:,:));
            r=corrcoef(temp,VEOG(:,goodepochs));
            VEOG_ICA_Corrs(ai)=abs(r(1,2)); clear temp;
        end
        bad_VEOG_ICAs=find(abs(zscore(VEOG_ICA_Corrs))>3);
        if isempty(bad_VEOG_ICAs), bad_VEOG_ICAs=find(VEOG_ICA_Corrs==max(abs(VEOG_ICA_Corrs))); end % in case z-scores are too tightly distributed
    else
        bad_VEOG_ICAs=0;
    end
    
    % Bootstrap a blink template based on Gaussian distros around most frontopolar channels
    % Get the most FrontoPolar Sites
    for ai=1:dims(1), X(ai)=EEG.chanlocs(ai).X; end
    FrontoPolars=find(X==max(X));  clear X;
    % Make Gaussian Template - code taken from Mike X Cohen
    for fpi=1:length(FrontoPolars)
        e2use=FrontoPolars(fpi);
        eucdist=zeros(1,size(EEG.icawinv,1)); topocorr=zeros(1,size(EEG.icawinv,1));
        for chani=1:size(EEG.icawinv,1)
            eucdist(chani)=sqrt( (EEG.chanlocs(chani).X-EEG.chanlocs(e2use).X)^2 + (EEG.chanlocs(chani).Y-EEG.chanlocs(e2use).Y)^2 + (EEG.chanlocs(chani).Z-EEG.chanlocs(e2use).Z)^2 );
        end
        s=30;   template(fpi,:) = exp(- (eucdist.^2)/(2*s^2) );
    end
    template=mean(template,1);
    % Get each ICA topo correlation with this topo template
    for chani=1:size(EEG.icawinv,2)
        topocorr(chani) = corr(EEG.icawinv(:,chani),template');
    end
    % Select the max correlations
    bad_TEMPLATE_ICAs=find(abs(zscore(topocorr))>3);
    if isempty(bad_TEMPLATE_ICAs), bad_TEMPLATE_ICAs=find(abs(topocorr)==max(abs(topocorr))); end % in case z-scores are too tightly distributed

    % Aggregate all this
    bad_ICAs{1}=bad_ADJUST_ICAs;
    bad_ICAs{2}=bad_VEOG_ICAs;
    bad_ICAs{3}=bad_TEMPLATE_ICAs;
    bad_ICAs{4}=[sum(goodepochs),epochsneeded];
    
elseif Do_ICA~=1 && hasVEOG==1
    
    % Do Gratton Method
    EEG.data = gratton( EEG.data, VEOG(:,goodepochs), 200, 20 );  % Defaults for voltage (200 uV) and window (20 ms) | requires statistics toolbox
    bad_ICAs='No ICAs, Ran Gratton';

end

%% Show Stats

elapsed=toc;
pBAD_CHANS=(length(bad_chans{3})./dims(1))*100;
pBAD_EPOCHS=(sum(bad_epochs{3})./dims(3))*100;

% Show ERP and Topo after rejecting blink ICA, but don't actually remove that from the real EEG data
tempeeg=EEG; % archive real set 
EEG = pop_subcomp( EEG, bad_TEMPLATE_ICAs, 0); % remove TEMPLATE ICAs
PostFixERP=eegfilt(squeeze(mean(EEG.data(Vertex,:,:),3)),EEG.srate,[],20);  % Get ERP
PostFixERP=PostFixERP-repmat(mean(PostFixERP),1,length(PostFixERP)); % Ersatz Baseline 
PostFixTopo=squeeze(mean(mean(EEG.data(:,T1:T2,:),2),3));  % Topo w/o blinks
clear EEG; EEG=tempeeg; clear tempeeg;   % recover archive set for output
 
figure;
subplot(2,3,1)
pie([dims(1)-length(bad_chans{3}),length(bad_chans{3})],[0 1],{['Good=',num2str(dims(1)-length(bad_chans{3}))],['Bad=',num2str(length(bad_chans{3}))]})
title(['Subj: ',SubjID, ' Bad Chans']);
subplot(2,3,2)
pie([dims(3)-sum(bad_epochs{EP2REJ}),sum(bad_epochs{EP2REJ})],[0 1],{['Good=',num2str(dims(3)-sum(bad_epochs{EP2REJ}))],['Bad=',num2str(sum(bad_epochs{EP2REJ}))]})
title(['Subj: ',SubjID, ' Bad Epochs']);
subplot(2,3,3)
if Do_ICA==1
    text(.2, .90, ['Bad ADJUST ICAs: ',num2str(bad_ICAs{1})]);
    text(.2, .75, ['Bad VEOGcorr ICAs: ',num2str(bad_ICAs{2})]);
    text(.2, .60, ['Bad TEMPLATE ICAs: ',num2str(bad_ICAs{3})]);
    text(.2, .45, ['Epochs Needed for ICA: ',num2str(bad_ICAs{4}(2))]);
    text(.2, .30, ['Epochs in Dataset (good): ',num2str(bad_ICAs{4}(1))]);
    text(.2, .15, ['Mins Elapsed: ',num2str(elapsed/60)]);
else
    text(.2, .50, bad_ICAs);
    text(.2, .05, ['Mins Elapsed: ',num2str(elapsed/60)]);
end
set(gca,'visible','off');  
%
subplot(2,3,4)
hold on
topoplot(PreFixTopo,EEG.chanlocs);
title('Topo Before Fixes (300-400 ms)');
subplot(2,3,5)
hold on
topoplot(PostFixTopo,EEG.chanlocs);
title('Topo After Fixes (300-400 ms)');
subplot(2,3,6)
hold on
plot(EEG.times,PreFixERP,'r');
plot(EEG.times,PostFixERP,'b--');
legend({'Pre-Fixes','Post-Fixes'},'Location','SouthOutside');
title('ERP at Vertex (20 Hz Filter)');
% Save that shiznit
saveas(gcf, [SubjID,'_',TASK,'_APPLE.png'],'png');
close all;

% Save a map of the original ICAs
pop_selectcomps(EEG, [1:20] );
saveas(gcf, [SubjID,'_',TASK,'_APPLE_ICAs.png'],'png');
close all;


% Plot percent variance accounted for
try
    [PV.pvaf,PV.pvafs,PV.vars] = eeg_pvaf(EEG,[],'plot','off');
    PV.pv_each(1)=PV.pvaf(1);
    for ci=2:20; PV.pv_each(ci)=PV.pvaf(ci)-PV.pvaf(ci-1); end
    figure; bar(PV.pv_each(1:20),'w'); set(gca,'xtick',1:20);
    saveas(gcf, [SubjID,'_',TASK,'_PVs.png'],'png');
catch  
    
end

function [lengths] = min_z_JFC(list_properties,rejection_options)
if (~exist('rejection_options','var'))
    rejection_options.measure=ones(1,size(list_properties,2));
    rejection_options.z=3*ones(1,size(list_properties,2));
end

rejection_options.measure=logical(rejection_options.measure);
zs=list_properties-repmat(mean(list_properties,1),size(list_properties,1),1);
zs=zs./repmat(std(zs,[],1),size(list_properties,1),1);
zs(isnan(zs))=0;
%all_l = abs(zs) > repmat(rejection_options.z,size(list_properties,1),1);
%lengths = any(all_l(:,rejection_options.measure),2);

lengths = abs(zs) > repmat(rejection_options.z,size(list_properties,1),1);


