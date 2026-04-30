function [alg, ode] = cooler(v, par,itg,dec, alpha , d )
    % Cooler with condensation of ammonia and a fixed duty
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
    % par: Parameters of the unit:
    % .Q: Heat transfer duty [W]
    % .Cp: Heat capacity of gas [J/kmol,K]
    % .Hvap Heat of vaporization [J/kmol]
    % itg: Intiger values, which should include
    % .k Total number of variables per stream
    % .s Total number of species in each stream
    % dec: Decision variable
    % .tempController2 Activate temperature controller
    % .dec.Tsep Perfect control of Tsep
    % alpha Separation ratio calculated in the separator
    % d: Disturbances
    % .d_Q2 Disturbance in heat duty [W]
    % The output structures are
    % ode Differential equations
    % alg Algebraic equations
    % Definition of multiples of itg.k
    k1 = itg.k;
    % Rearrangement of the variables
    nin = v(1);
    Pin = v(2);
    Tin = v(3);
    xin = v(4:itg.k);
    nout = v(k1+1);
    Pout = v(k1+2);
    Tout = v(k1+3);
    xout = v(k1+4:end);
    % Mole balance:s
    dndt = nout-nin;
    % Pressure balance:
    dPdt = Pin-Pout;
    % Energy balance:
    if dec.tempController2 == 1
        dTdt = Tout - dec.Tsep;
    else
        dTdt = ((par.Q+d.d_Q2) - nin*par.Cp*(Tout-Tin)-nin*(-par.Hvap)*alpha);
    end
    % Component balance:
    dxdt = xout-xin;
    alg = vertcat(dndt{:}, dPdt{:}, dTdt{:}, dxdt{:});
    ode = [];
end