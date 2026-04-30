function [alg,ode] = reactorBed(v, par, itg )
    % Reactor bed
    % The state variables are defined as
    % n = v(1,:): Flow [kmol/s]
    % P = v(2,:): Pressure [bar]
    % T = v(3,:): Temperature [C]
    % x = v(4:k1): Composition [-]
    % The necessary structures are defined as
    % par: Parameters ot the unit, must include:
    % ns Number of sections (CSTRs)
    % stoi Stochiometric coeff vector
    % Vbed Volume of bed [m3]
    % rhocat Bulk density of catalyst [kg/m3]
    % Cp Heat capacity of gas [J/kmol, K]
    % Cpcat Heat capacity of catalyst [J/ kg cat, K]
    % dHrx Heat of reaction [J/ kmol ]
    % + reactionRate parameters (see function reactionRate)
    % itg: Intiger values, which should include
    % .k Total number of variables per stream
    % .s Total number of species in each stream
    % The output structures are
    % ode Differential equations
    % alg Algebraic equations
    %Reshape input
    v = reshape(v,itg.k,par.ns+1);
    %Rearrangement of variables
    n = v(1,:);
    P = v(2,:);
    T = v(3,:);
    x = v(4:7,:);
    %Calculation of catalyst mass in each section
    mcat = (par.Vbed/par.ns)*par.rhocat; %m3*kg/m3
    %Predefinition of cells
    dndt = cell(par.ns,1);
    dPdt = cell(par.ns,1);
    dxdt_H2 = cell(par.ns,1);
    dxdt_N2 = cell(par.ns,1);
    dxdt_NH3 = cell(par.ns,1);
    dxdt_inert = cell(par.ns,1);
    dTdt = cell(par.ns,1);
    for j=1:par.ns
        r = reactionRate(T(j+1),P(j+1),x(1:end,j+1),par);
        dndt{j} = (n(j+1) - (n(j) + sum(par.stoi)*r*mcat));
        dPdt{j} = (P(j+1)- P(j));
        dxdt_H2{j} = (n(j+1)*x(1,j+1)-(n(j)*x(1,j)+mcat*r*par.stoi(1)));
        dxdt_N2{j} = (n(j+1)*x(2,j+1)-(n(j)*x(2,j)+mcat*r*par.stoi(2)));
        dxdt_NH3{j} = (n(j+1)*x(3,j+1)-(n(j)*x(3,j)+mcat*r*par.stoi(3)));
        dxdt_inert{j} = n(j+1)*x(4,j+1)-(n(j)*x(4,j));
        dTdt{j} = (par.Cp*(n(j)*T(j)-n(j+1)*T(j+1)) + r*mcat*par.dHrx)/(mcat*par.Cpcat);
    end
    alg = vertcat(dndt{:}, dPdt{:}, dxdt_H2{:},dxdt_N2{:},dxdt_NH3{:},dxdt_inert{:});
    ode = vertcat(dTdt{:});
end
