function [xfinal,ffinal,exitflag,xstart] = SETUP_RLMEM(BEH4MODEL,Modeltype)

options=optimset('display','off','LargeScale','off');
RMRUNS=10;

if strmatch(Modeltype,'Foil')
    FOIL_LLE=abs(sum(log(.5*ones(1,size(BEH4MODEL,1)))));
    xfinal=1; ffinal=FOIL_LLE;  exitflag=1; xstart=1;
    
elseif strmatch(Modeltype,'Vanilla')
    init_params = [.5,.5]';
    lower_limits = [-inf;-inf];
    upper_limits = [+inf;+inf];
    [xfinal,ffinal,exitflag,xstart] = rmsearch(@(params) Vanilla(params, BEH4MODEL),...
        'fminsearch',init_params,lower_limits,upper_limits,'Options',options,'InitialSample',RMRUNS);

elseif strmatch(Modeltype,'Vanilla2')
    init_params = [.5,.5,.5]';
    lower_limits = [-inf;-inf;-inf];
    upper_limits = [+inf;+inf;+inf];
    [xfinal,ffinal,exitflag,xstart] = rmsearch(@(params) Vanilla2(params, BEH4MODEL),...
        'fminsearch',init_params,lower_limits,upper_limits,'Options',options,'InitialSample',RMRUNS);

 
end





%%

