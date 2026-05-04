function [PE,Q,Params] = SETUP_RLMEM_Recover(BEH4MODEL,Modeltype,params_out)


if strmatch(Modeltype,'Foil')
    [PE]=NaN;  [Q]=NaN;
    Params=NaN;
    
elseif strmatch(Modeltype,'Vanilla')
    [PE,Q] = Vanilla_Recover(params_out,BEH4MODEL);
    alfa=1./(1+exp(-params_out(1)));
    beta=exp(-params_out(2));
    Params=[alfa,beta];
    
elseif strmatch(Modeltype,'Vanilla2')
    [PE,Q] = Vanilla2_Recover(params_out,BEH4MODEL);
    alfa_G=1./(1+exp(-params_out(1)));
    alfa_L=1./(1+exp(-params_out(2)));
    beta=exp(-params_out(3));
    Params=[alfa_G,alfa_L,beta];
    
end



