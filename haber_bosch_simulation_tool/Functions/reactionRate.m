function r = reactionRate(T,P,x,par)
    % This function calculates the reation rate of ammonia in kg NH3/kmol cat,s.
    % The necessary input structures are defined as
    % T Temperature of the stage [C]
    % P Pressure of the stage [bar]
    % x Molar fraction of the stage [-]
    % par Structure containing parameters for the reaction rate constants
    % Afor Arrhenius factor, forward
    % Abac Arrhenius factor, backward
    % Eafor Activation Energy, forward [J/mol]
    % Eabac Activation Energy, backward [J/mol]
    % R Gas constant [J/mol, K]
    % Calculation of the reaction rate constants
    k1 = par.Afor*exp(-par.Eafor./(par.R*(T+273.15))); % Forward reaction rate constant
    k2 = par.Abac*exp(-par.Eabac./(par.R*(T+273.15))); % Backward reaction rate constant
    pH2 = x(1)*P;
    pN2 = x(2)*P;
    pNH3 = x(3)*P;
    % Calculation of the reaction rate and transformation of it
    r = k1*pN2*pH2^1.5/pNH3 - k2*pNH3/pH2^1.5; % [kmol N2/ m3 cat, h]
    r = r/3600/par.rhocat; % [kmol N2/ kg cat, s]
    r = 4.75*r; % catalyst activity
end
