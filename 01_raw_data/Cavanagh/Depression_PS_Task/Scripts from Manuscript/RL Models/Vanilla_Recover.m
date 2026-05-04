function [PE,Q]=Vanilla_Recover(params,BEH4MODEL)
% Vanilla RL with 1 l-rate and smx beta

alfa=1./(1+exp(-params(1)));
beta=exp(-params(2));      

% Initialize     
Q=.5*ones(3,2);  % [A,B ; C,D ; E,F];

for t=1:size(BEH4MODEL,1)
    
    % Cue
    Cue=BEH4MODEL(t,2);   % 1=AB, 2=CD, 3=EF
    % Action Selection
    Action=BEH4MODEL(t,3); % 1=optimal (A,C,E), 0=suboptimal (B,D,F)
    % Reinforcement
    FB=BEH4MODEL(t,4);     % 1=Rew, 0=Pun 
    
    % PE
    PE(t,1)=FB-Q(Cue,2-Action);
    
    % Softmax
    PA(t,1)=exp(beta*Q(Cue,2-Action))/sum(exp(beta*Q(Cue,:)));
    
    % Update Q Value
    Q_time(t,1)=Q(Cue,2-Action) + alfa*(PE(t,1));
    Q(Cue,2-Action)=Q(Cue,2-Action) + alfa*(PE(t,1));
    
end


%%





