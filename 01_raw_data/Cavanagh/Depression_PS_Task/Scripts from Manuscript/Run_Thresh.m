function Corrected_P=Run_Thresh(P_2D)

ThisThreshold=500; 
Corrected_P=zeros(50,751);
P=squeeze(P_2D);  
l=bwlabel(P_2D);
lmax=max(reshape(l,1,50*751));
if lmax>0
    for ei=1:lmax
        [row,col] = find(l == ei);
        for clusti=1:size(row,1);
            temp(clusti) = abs(P_2D(row(clusti),col(clusti)));
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

