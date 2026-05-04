function [Corrected_P] = Run_Thresh_2D(TEMP1,TEMP2,ttesttype)

SHUFFLES=5000;
for shuffi=1:SHUFFLES

    tempAB=cat(1,TEMP1,TEMP2);
    idx=shuffle([ones(1,25),zeros(1,25)]);
    A=tempAB(idx==1,:,:);
    B=tempAB(idx==0,:,:);
    
    if strmatch(ttesttype,'between')
         [H,P,CI,STATS]=ttest2(A,B);
    elseif strmatch(ttesttype,'within')
         [H,P,CI,STATS]=ttest(A,B);
    end
    P(P<=.05)=NaN; P(P>.05)=0; P(isnan(P))=1;
    
    P=squeeze(P); STATS.tstat=squeeze(STATS.tstat);
    l=bwlabel(P);
    dims=size(P);
    lmax=max(reshape(l,1,dims(1)*dims(2)));
    if lmax>0
        for ei=1:lmax
            [row,col] = find(l == ei);
            for clusti=1:size(row,1);
                temp(clusti) = abs(STATS.tstat(row(clusti),col(clusti)));
            end
            tempthresh(ei) = sum(temp);   clear temp;
        end
        THRESH(shuffi) = max(tempthresh);
        clear row col tempthresh ;
    else
        THRESH(shuffi) = 0;
    end
    clear H CI P STATS temp* A B idx  l dims lmax;
    
end
THRESH=sort(THRESH);
ThisThreshold=THRESH(end-SHUFFLES*.05);

% NOW Run 1D size of effects
if strmatch(ttesttype,'between')
    [H,P,CI,STATS]=ttest2(TEMP1,TEMP2);
elseif strmatch(ttesttype,'within')
    [H,P,CI,STATS]=ttest(TEMP1,TEMP2);
end
P(P<=.05)=NaN; P(P>.05)=0; P(isnan(P))=1;

Corrected_P=zeros(50,751);
P=squeeze(P); STATS.tstat=squeeze(STATS.tstat);
l=bwlabel(P);
dims=size(P);
lmax=max(reshape(l,1,dims(1)*dims(2)));
if lmax>0
    for ei=1:lmax
        [row,col] = find(l == ei);
        for clusti=1:size(row,1);
            temp(clusti) = abs(STATS.tstat(row(clusti),col(clusti)));
        end
        if sum(temp) > ThisThreshold;
             for clusti=1:size(row,1);
                Corrected_P(row(clusti),col(clusti))=1;
             end
        end
        clear temp row col;
    end
end
clear H CI P STATS temp* A B idx  l dims lmax;

clear THRESH ThisThreshold

