function [alg,ode] = makeupMixer(v,itg,d)
    % This function mix two streams into one stream
    % The variables are defined as
    % v(1): Inlet flow stream [kmol/s]
    % v(2): Inlet pressure stream [bar]
    % v(3): Inlet temperature stream [C]
    % v(4:k1): Inlet composition stream [-]
    % v(k1+1): Inlet flow stream [kmol/s]
    % v(k1+2): Inlet pressure stream [bar]
    % v(k1+3): Inlet temperature stream [C]
    % v(k1+4:k2): Inlet composition stream [-]
    % v(k2+1): Outlet flow stream [kmol/s]
    % v(k2+2): Outlet pressure stream [bar]
    % v(k2+3): Outlet temperature stream [C]
    % v(k2+4:end): Outlet composition stream [-]
    % The necessary input structures are defined as
    % itg: Intiger values
    % .k Total number of variables per stream
    % d Disturbances
    % .d_P Pressure disturbance
    % The output structures are
    % ode Differential equations
    % alg Algebraic equations
    % Definition of multiples of itg.k
    k1 = itg.k;
    k2 = 2*k1;
    % Rearrangement of variables
    nin1 = v(1);
    Pin1 = v(2);
    Tin1 = v(3)+273.15;
    xin1 = v(4:itg.k);
    nin2 = v(k1+1);
    Pin2 = v(k1+2);
    Tin2 = v(k1+3)+273.15;
    xin2 = v(k1+4:k2);
    nout = v(k2+1);
    Pout = v(k2+2);
    Tout = v(k2+3)+273.15;
    xout = v(k2+4:end);
    % Mole fractions:
    d_nin1 = nin1/(nout); %stream 1/total stream
    d_nin2 = nin2/(nout); %stream 2/total stream
    % Mole balance:
    dndt = nout - (nin1+nin2);
    % Energy balance:
    dTdt = Tout - (d_nin1*Tin1+ d_nin2*Tin2);
    % Component balance:
    dxdt = (xout - (d_nin1*xin1+ d_nin2*xin2));
    % Pressure neutrality:
    dPdt{1} = Pout - (Pin1+d.d_P);
    dPdt{2} = Pout - (Pin2+d.d_P);
    alg = vertcat(dndt, dPdt{:} , dTdt, dxdt{:});
    ode = [];
end