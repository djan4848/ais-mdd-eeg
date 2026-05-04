%% TF
rx_dep_anx=repmat(corr( TheseData(DEP,DEPVAR) , TheseData(DEP,ANXVAR)  ,'type','Spearman'),50*751,1);
frex=logspace(.01,1.7,50);

TFchani=1;

TFt1=[200,200];
TFt2=[400,400];
TFf1=[19,8];  % 4, 2
TFf2=[27,14];  % 8, 3

figure; hold on

% TF
for paneli=1:2
    
    subplot(2,2,paneli); hold on
    imagesc(tx2disp,[],squeeze( mean(MEGA_PWR_CORR{TFchani}(DEP,:,:,paneli),1) ) ); axis xy
    plot([0 0],[1 50],'k:');
    plot([TFt1(paneli), TFt1(paneli)],[TFf1(paneli), TFf2(paneli)],'k:'); 
    plot([TFt2(paneli), TFt2(paneli)],[TFf1(paneli), TFf2(paneli)],'k:');
    plot([TFt1(paneli), TFt2(paneli)],[TFf1(paneli), TFf1(paneli)],'k:'); 
    plot([TFt1(paneli), TFt2(paneli)],[TFf2(paneli), TFf2(paneli)],'k:');
    set(gca,'clim',[-.1 .1],'xlim',[-500,1000],'ylim',[1 50],'YTick',1:4:length(frex),'YTickLabel',round(frex(1:4:end)));
    
end

% Corr with Symptoms
for paneli=1:2
    
    subplot(2,2,paneli+2); hold on
    TEMP=squeeze(MEGA_PWR_CORR{TFchani}(DEP,:,:,paneli));
    dims=size(TEMP);
    TEMP=reshape(TEMP,dims(1),dims(2)*dims(3));
    
    % Compute corrs with each var
    ToCorr=TheseData(DEP,DEPVAR);
    [rho_dep,p_dep]=corr( TEMP , ToCorr  ,'type','Spearman');
    p_dep(p_dep<=.05)=NaN; p_dep(p_dep>.05)=0; p_dep(isnan(p_dep))=1; 
    rho_dep_2D=reshape(rho_dep,dims(2),dims(3));
    p_dep_2D=reshape(p_dep,dims(2),dims(3));
    corrected_p_dep=Run_Thresh(p_dep_2D);

    ToCorr=TheseData(DEP,ANXVAR);
    [rho_anx,p_anx]=corr( TEMP , ToCorr  ,'type','Spearman');
    p_anx(p_anx<=.05)=NaN; p_anx(p_anx>.05)=0; p_anx(isnan(p_anx))=1; 
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
    TF_ROI(:,paneli)=squeeze( mean(mean( MEGA_PWR_CORR{TFchani}(:,TFf1(paneli):TFf2(paneli),...
        find(tx2disp==TFt1(paneli)):find(tx2disp==TFt2(paneli)),paneli)  ,3),2) );
end

GROUP=DEP;
V1=TF_ROI(GROUP,1);    % Main correlate for beh (Punishment Theta)
V1_C=TF_ROI(GROUP,2);  % This one is the contrast (Reward Delta)

figure; hold on

subplot(4,4,[1,2,5,6]); hold on
V2=TheseData(GROUP,17)-TheseData(GROUP,18);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'k'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[-1 1],'ylim',[-1 1]);
title('Go-NoGo')

subplot(4,4,9); hold on
V2=TheseData(GROUP,18);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'r'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[-1 1],'ylim',[0 1]);
title('NoGo')
% ---
subplot(4,4,10);
V2=TheseData(GROUP,17);
[rho,rho_p]=corr(V1,V2,'type','Spearman');
scatter(V1,V2,'g'); lsline
text([.1],[.6],['rho=',num2str(rho),' p=',num2str(rho_p)],'sc');
[rho_C,rho_C_p]=corr(V1_C,V2,'type','Spearman');
[menghyp,mengp,mengzscore] = mengz_JFC(rho,rho_C,r12,n);
text([.1],[.4],['rho_C=',num2str(rho_C),' p=',num2str(rho_C_p)],'sc');
text([.1],[.2],['z=',num2str(mengzscore),' p=',num2str(mengp)],'sc');
set(gca,'xlim',[-1 1],'ylim',[0 1]);
title('Go')

