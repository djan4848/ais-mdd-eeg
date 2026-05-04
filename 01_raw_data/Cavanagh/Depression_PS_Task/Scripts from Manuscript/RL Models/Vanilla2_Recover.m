function [PE,Q]=Vanilla2_Recover(params,BEH4MODEL)
% Vanilla RL with 1 l-rate and smx beta

alfa_G=1./(1+exp(-params(1)));
alfa_L=1./(1+exp(-params(2)));
beta=exp(-params(3));          

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
    if FB==1,
        Q_time(t,1)=Q(Cue,2-Action) + alfa_G*(PE(t,1));
        Q(Cue,2-Action)=Q(Cue,2-Action) + alfa_G*(PE(t,1));
    elseif FB==0,
        Q_time(t,1)=Q(Cue,2-Action) + alfa_L*(PE(t,1));
        Q(Cue,2-Action)=Q(Cue,2-Action) + alfa_L*(PE(t,1));
    end
    
end


%%





