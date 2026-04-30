function [alg, ode] = compressor(v, par,itg, d )
    % Piston compressor
    % The variables are defined as
    % v(1): Inlet flow [kmol/s]
    % v(2): Inlet pressure [bar]
    % v(3): Inlet temperature [C]
    % v(4:k1): Inlet composition [-]
    % v(k1+1): Outlet flow [kmol/s]
    % v(k1+2): Outlet pressure [bar]
    % v(k1+3): Outlet temperature [C]
    % v(k1+4:k2): Outlet composition [-]
    % The necessary input structures are defined as
    % par: Parameters of unit
    % .Cp Heat capacity [J/kmol]
    % .R Gas constant [J/kmol K]
    % .W Compressor duty [W]
    % .nc Compressor efficiency [-]
    % itg: Intiger values
    % .k Total number of variables per stream
    % d: Disturbances
    % .d_W Disturbance in compressor duty [W]
    % The output structures are
    % ode Differential equations
    % alg Algebraic equations
    % Definition of multiples of itg.k
    k1 = itg.k;
    % Rearrangement and scaling of variables
    nin = v(1);
    Pin = v(2);
    Tin = v(3)+273.15;
    xin = v(4:itg.k);
    nout = v(k1+1);
    Pout = v(k1+2);
    Tout = v(k1+3)+273.15;
    xout = v(k1+4:end);
    % Mole balance:
    dndt = nout-nin;
    % Pressure balance:
    dPdt = ((par.Wcomp+d.d_W)*par.nc - nin*par.Cp*Tin*((Pout/Pin)^(par.R/par.Cp)-1));
    % Energy balance:
    dTdt = (Tout - Tin*(1+((Pout/Pin)^(par.R/par.Cp)-1)/par.nc));
    % Component balance:
    dxdt = xout - xin;
    alg = vertcat(dndt{:}, dPdt{:}, dTdt{:}, dxdt{:});
    ode = [];
end