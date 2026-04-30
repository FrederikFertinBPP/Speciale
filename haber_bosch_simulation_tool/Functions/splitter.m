function [Ceq,ddt] = splitter(v, par, itg, d )
    % This function split one stream into 3 streams
    % The variables are defined as
    % v(1): Inlet flow [kmol/s]
    % v(2): Inlet pressure [bar]
    % v(3): Inlet temperature [C]
    % v(4:k1): Inlet composition [-]
    % v(k1+1): Outlet flow [kmol/s]
    % v(k1+2): Outlet pressure [bar]
    % v(k1+3): Outlet temperature [C]
    % v(k1+4:k2): Outlet composition [-]
    % v(k2+1): Outlet flow [kmol/s]
    % v(k2+2): Outlet pressure [bar]
    % v(k2+3): Outlet temperature [C]
    % v(k2+4:k3): Outlet composition [-]
    % v(k3+1): Outlet flow [kmol/s]
    % v(k3+2): Outlet pressure [bar]
    % v(k3+3): Outlet temperature [C]
    % v(k3+4:k4): Outlet composition [-]
    % The necessary structures are defined as
    % par: Parameters of the unit, must include:
    % u1 split ratio of nout1
    % u2 split ratio of nout2
    % scl: Scaling variables
    % itg: Intiger values, which should include
    % .k Total number of variables per stream
    % .s Total number of species in each stream
    % d: Disturbances
    % .d_u1 Disturbance in u1 [-]
    % .d_u2 Disturbance in u2 [-]
    % The output structures are
    % ode Differential equations
    % alg Algebraic equations
    % Definition of multiples of itg.k
    k1 = itg.k;
    k2 = 2*k1;
    k3 = 3*k1;
    k4 = 4*k1;
    % Rearrangement of variables
    nin = v(1);
    Pin = v(2);
    Tin = v(3);
    xin = v(4:itg.k);
    nout = v(k1+1);
    Pout = v(k1+2);
    Tout = v(k1+3);
    xout = v(k1+4:k2);
    nout2 = v(k2+1);
    Pout2 = v(k2+2);
    Tout2 = v(k2+3);
    xout2 = v(k2+4:k3);
    nout3 = v(k3+1);
    Pout3 = v(k3+2);
    Tout3 = v(k3+3);
    xout3 = v(k3+4:k4);
    u1 = par.u1+d.d_u1;
    u2 = par.u2+d.d_u2;
    % Mole balance
    dNdt{1} = nout - u1*nin;
    dNdt{2} = nout2 - u2*nin;
    dNdt{3} = nout3 - (1-u1-u2)*nin;
    % Pressure neutrality
    dPdt{1} = (Pout-Pin);
    dPdt{2} = (Pout2-Pin);
    dPdt{3} = (Pout3-Pin);
    % Temperature neutrality
    dTdt{1} = (Tout-Tin);
    dTdt{2} = (Tout2-Tin);
    dTdt{3} = (Tout3-Tin);
    % Composition balance
    dxdt{1} = xout-xin;
    dxdt{2} = xout2-xin;
    dxdt{3} = xout3-xin;
    Ceq = vertcat(dNdt{:}, dPdt{:}, dTdt{:}, dxdt{:});
    ddt = [];
end
