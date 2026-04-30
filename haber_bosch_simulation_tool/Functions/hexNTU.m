function [alg, ode] = hexNTU(v, par, itg )
    % NTU heat exchanger
    % The variables are defined as
    % v(1): Inlet flow hot stream [kmol/s]
    % v(2): Inlet pressure hot stream [bar]
    % v(3): Inlet temperature hot stream [C]
    % v(4:k1): Inlet composition hot stream [-]
    % v(k1+1): Outlet flow hot stream [kmol/s]
    % v(k1+2): Outlet pressure hot stream [bar]
    % v(k1+3): Outlet temperature hot stream [C]
    % v(k1+4:k2): Outlet composition hot stream [-]
    % v(k2+1): Inlet flow cold stream [kmol/s]
    % v(k2+2): Inlet pressure cold stream [bar]
    % v(k2+3): Inlet temperature cold stream [C]
    % v(k2+4:k3): Inlet composition cold stream [-]
    % v(k3+1): Outlet flow cold stream [kmol/s]
    % v(k3+2): Outlet pressure cold stream [bar]
    % v(k3+3): Outlet temperature cold stream [C]
    % v(k3+4:k4): Outlet composition cold stream [-]
    % The necessary structures are defined as
    % par: Parameters ot the unit, must include:
    % .U: Heat transfer coefficient [W/m2, K]
    % .A: Heat transfer area [m2]
    % .Cph: Molar heat capacity hot stream [J/kmol,K]
    % .Cpc: Molar heat capacity cold stream [J/kmol,K]
    % The output structures are
    % ode Differential equations
    % alg Algebraic equations
    % Definition of multiples of itg.k
    k1 = itg.k;
    k2 = 2*itg.k;
    k3 = 3*itg.k;
    k4 = 4*itg.k;
    
    % Rearrangement of the variables
    nin_h = v(1);
    Pin_h = v(2);
    Tin_h = v(3);
    xin_h = v(4:itg.k);
    
    nout_h = v(k1+1);
    Pout_h = v(k1+2);
    Tout_h = v(k1+3);
    xout_h = v(k1+4:k2);
    
    nin_c = v(k2+1);
    Pin_c = v(k2+2);
    Tin_c = v(k2+3);
    xin_c = v(k2+4:k3);
    
    nout_c = v(k3+1);
    Pout_c = v(k3+2);
    Tout_c = v(k3+3);
    xout_c = v(k3+4:k4);
    
    % Heat capacity ratio
    Cstar = (nin_c*par.Cpc)/(nin_h*par.Cph);
    % Note: the cold stream has the smallest heat capacity rate
    % Number of transfer units
    NTU = (par.U*par.A)/(nin_c*par.Cpc);
    % Effectiveness
    E = (1-exp(-NTU*(1-Cstar)))/(1-Cstar*exp(-NTU*(1-Cstar)));
    % Maximum heat transfer
    Qmax = nin_c*par.Cpc*(Tin_h-Tin_c);
    % Actual heat transfer
    Q = E*Qmax;
    % Mole balance:
    dndt{1} = nout_h-nin_h;
    dndt{2} = nout_c-nin_c;
    % Pressure balance:
    dPdt{1} = Pout_h-Pin_h;
    dPdt{2} = Pout_c-Pin_c;
    % Energy balance:
    dTdt{1} = (Q-nin_h*par.Cph*(Tin_h-Tout_h));
    dTdt{2} = (Q-nin_c*par.Cpc*(Tout_c-Tin_c));
    % Component balance:
    dxdt{1} = xout_h-xin_h;
    dxdt{2} = xout_c-xin_c;
    alg = vertcat(dndt{:}, dPdt{:}, dTdt{:}, dxdt{:});
    ode = [];
end
