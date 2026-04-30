function [alg, ode] = simpleHEX(v, par,itg, d)
    % Simple heat-exchanger
    % The variables are defined as
    % v(1): Inlet flow [kmol/s]
    % v(2): Inlet pressure [bar]
    % v(3): Inlet temperature [C]
    % v(4:k1): Inlet composition [-]
    % v(k1+1): Outlet flow [kmol/s]
    % v(k1+2): Outlet pressure [bar]
    % v(k1+3): Outlet temperature [C]
    % v(k1+4:k2): Outlet composition [-]
    % The necessary structures are defined as
    % par: Parameters ot the unit, must include:
    % .Q: Heat duty [W]
    % .Cph: Heat capacity [J/kmol,K]
    % itg: Intiger values
    % .k Total number of variables per stream
    % d: Disturbances
    % .d_Q1 Disturbance in heat duty [W]
    % The output structures are
    % ode Differential equations
    % alg Algebraic equations
    % Definition of multiples of itg.k
    k1 = itg.k;
    k2 = 2*itg.k;
    % Rearrangement of the variables
    nin = v(1);
    Pin = v(2);
    Tin = v(3);
    xin = v(4:itg.k);
    nout = v(k1+1);
    Pout = v(k1+2);
    Tout = v(k1+3);
    xout = v(k1+4:k2);
    %Mole balance:
    dndt{1} = nout-nin;
    %Pressure balance:
    dPdt{1} = Pout-Pin;
    %Energy balance:
    dTdt{1} = ((par.Q+d.d_Q1)-nin*par.Cp*(Tout-Tin));
    %Component balance:
    dxdt{1} = xout-xin;
    alg = vertcat(dndt{:}, dPdt{:}, dTdt{:}, dxdt{:});
    ode = [];
end
