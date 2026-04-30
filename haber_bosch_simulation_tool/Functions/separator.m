function [alg, ode] = separator(v, par, itg ,dec, alpha, d)
    % Separator
    % The variables are defined as
    % v(1): Inlet flow [kmol/s]
    % v(2): Inlet pressure [bar]
    % v(3): Inlet temperature [C]
    % v(4:k1): Inlet composition [-]
    % v(k1+1): Product flow [kmol/s]
    % v(k1+2): Product pressure [bar]
    % v(k1+3): Product temperature [C]
    % v(k1+4:k2): Product composition [-]
    % v(k2+1): Purge flow [kmol/s]
    % v(k2+2): Purge pressure [bar]
    % v(k2+3): Purge temperature [C]
    % v(k2+4:k3): Purge composition [-]
    % v(k3+1): Outlet flow [kmol/s]
    % v(k3+2): Outlet pressure [bar]
    % v(k3+3): Outlet temperature [C]
    % v(k3+4:end): Outlet composition [-]
    % The necessary structures are defined as
    % par: Parameters ot the unit, must include:
    % .Kvlv Purge valve opening [-]
    % .Khx Pressure resistance [kmol/bar]
    % .Vtot Total volume of loop [m3]
    % .R Gas constant [m3 bar/kmol, K]
    % .A Antoine Equation Parameter
    % .B Antoine Equation Parameter
    % .C Antoine Equation Parameter
    % .H1 Henry's constant polynomial
    % .H2 Henry's constant polynomial
    % .H3 Henry's constant polynomial
    % itg: Intiger values
    % .k Total number of variables per stream
    % dec: Decision variable
    % .dynamicPressure
    % alpha Separation ratio
    % d: Disturbances
    % .d_Kvlv Disturbance in purge valve opening
    % The output structures are
    % ode Differential equations
    % alg Algebraic equations
    % Definition of multiples of itg.k
    k1 = itg.k;
    k2 = 2*itg.k;
    k3 = 3*itg.k;
    % Rearrangement of the variables
    nin = v(1);
    Pin = v(2);
    Tin = v(3)+273.15;
    xin = v(4:itg.k);
    nprod = v(k1+1);
    Pprod = v(k1+2);
    Tprod = v(k1+3)+273.15;
    xprod = v(k1+4:k2);
    npurge = v(k2+1);
    Ppurge = v(k2+2);
    Tpurge = v(k2+3)+273.15;
    xpurge = v(k2+4:k3);
    nout = v(k3+1);
    Pout = v(k3+2);
    Tout = v(k3+3)+273.15;
    xout = v(k3+4:end);
    % Raoult's law for NH3
    P_NH3 = (1.01325)*(par.A + par.B*Tin + par.C*Tin^2 + par.D*Tin^3 + par.E*Tin^4);
    % Henry's law for H2, N2 and inert:
    H = (1.01325)*exp(par.H1 + par.H2/Tin + par.H3/Tin^2);
    % Calculation of equilibrium constant:
    Keq = [H(1); H(2); P_NH3; H(3)];
    % Mole balance:
    dndt{1} = nin*xin - nprod*xprod - (nin-nprod)*xpurge;
    dndt{2} = npurge - (par.Kvlv+d.d_Kvlv)*sqrt(Pprod-Ppurge);
    dndt{3} = alpha - nprod/(nin*xin(3));
    % Pressure balance:
    dPdt{1} = Ppurge - par.Ppurge;
    dPdt{2} = Pout - Pprod;
    dPdt{3} = nin-par.Khx*sqrt(Pin-Pprod);
    if dec.dynamicPressure == 1
        ode =(((par.R*Tin)/par.Vtot)*(par.Khx*sqrt(Pin-Pprod)-nprod-npurge-nout));
    else
        dndt{4} = nout - (nin-nprod-npurge);
        ode = [];
    end
    % Energy balance
    dTdt{1} = Tprod - Tin;
    dTdt{2} = Tpurge - Tin;
    dTdt{3} = Tout - Tin;
    % Component balance:
    dxdt{1} = 1 - sum(xprod);
    dxdt{2} = xout - xpurge ;
    dxdt{3} = xpurge*Pprod - Keq.*xprod;
    alg = vertcat(dndt{:}, dPdt{:}, dTdt{:}, dxdt{:});
end