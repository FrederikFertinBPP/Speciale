% This script runs the ammonia loop function
% Clear workspace and close windows
clc; clear; close all;

% Definition of the current path folder
FileName = mfilename('fullpath');
[directory,~,~] = fileparts(FileName);
[dirmain,~,~] = fileparts(directory);

% Import of CASADI
addpath(['/CasADi'])
addpath([directory '/Functions'])
% addpath([dirmain '/matlab2tikz/src'])
import casadi.*

% Load parameters
parAmmoniaLoop;

% Definition of decision variables
dec.reactor = true; % Run only the reactor section
dec.wo_recycle = true; % Run without material recycle
dec.dynamicPressure = true; % Include dynamic pressure in separator
dec.dynamic = true; % Run dynamic simulations
dec.vanHeerden = true*dec.reactor; % Van Heerden analysis.
% Note: Requires that
% dec.reactor = true

dec.makeupController = true; % Makeup flow controller
dec.tempController1 = true; % Control inlet temperature to 1st bed

dec.VPC = false*dec.tempController1;% Valve position control
dec.tempController2 = true; % Control cooler outlet temperature
dec.Tsep = 12.5*dec.tempController2;%Set separator temperature
dec.tempController3 = true; % Control heater outlet temperature
dec.pressureController = true; % Controller pressure
dec.plot = true; % Make plots

% Make-up stream variables:
% Note: The make-up flow is calculated in the model.
f = zeros(7,1);
f(1) = 0; % Make-up flow [kmol/s]
f(2) = 40; % Pressure [bar]
f(3) = 25; % Temperature [C]
f(4) = 0.7479; % Molar fraction H2 [-]
f(5) = 0.2493; % Molar fraction N2 [-]
f(6) = 0; % Molar fraction NH3 [-]
f(7) = 0.0028; % Molar fraction inert [-]

% Recycle stream variables:
% Note: This stream is used if you run the loop without material recycle
% (dec.wo_recycle = true)
r = zeros(7,1);
r(1) = 0.3213; % Recycle flow [kmol/s]
r(2) = 150.1169; % Pressure [bar]
r(3) = 22.7665; % Temperature [C]
r(4) = 0.6976; % Molar fraction H2 [-]
r(5) = 0.2325; % Molar fraction N2 [-]
r(6) = 0.0541; % Molar fraction NH3 [-]
r(7) = 0.0158; % Molar fraction inert [-]

if dec.reactor == true
    % Definition of reactor feed variables
    % Note: This stream is used if you only run the reactor section
    f = zeros(7,1);
    f(1) = 0.4073; % Feed flow kmol/s
    f(2) = 150.2717; % Pressure [bar]
    f(3) = 129.9333; % Temperature [C]
    f(4) = 0.7050; % Mass fraction H2 [-]
    f(5) = 0.2398; % Mass fraction N2 [-]
    f(6) = 0.0351; % Mass fraction NH3 [-]
    f(7) = 0.0202; % Mass fraction inert [-]
    % Define number of algebraic variables in each unit
    nz.makeupComp = 0;
    nz.mix = 0;
    nz.heater = 0;
    nz.split = 21;
    nz.preheat = 20;
    nz.mix1 = 7;
    nz.reactor1 = par{ind.reactor1}.ns*6;
    nz.mix2 = 7;
    nz.reactor2 = (par{ind.reactor2}.ns-1)*6;
    nz.cooler = 0;
    nz.separator = 0;
    nz.recComp = 0;
    nz.alpha = 0;
    % Define number of differential variables in each unit
    nx.preheat = 1;
    nx.reactor1 = par{ind.reactor1}.ns;
    nx.reactor2 = par{ind.reactor2}.ns-nx.preheat;
    nx.separator = 0;
else
    % Define number of algebraic variables in each unit
    nz.makeupComp = 8;
    nz.mix = 14;
    nz.heater = 7;
    nz.split = 21;
    nz.preheat = 20;
    nz.mix1 = 7;
    nz.reactor1 = par{ind.reactor1}.ns*(itg.k-1);
    nz.mix2 = 7;
    nz.reactor2 = (par{ind.reactor2}.ns-1)*(itg.k-1);
    nz.cooler = 7;
    nz.separator = 21;
    nz.recComp = 0;
    nz.alpha = 1;
    % Define number of differential vari
    nx.preheat = 1;
    nx.reactor1 = par{ind.reactor1}.ns;
    nx.reactor2 = par{ind.reactor2}.ns-nx.preheat;
    nx.separator = 0;
    if dec.dynamicPressure == 1
        nz.separator = 20;
        nx.separator = 1;
    end
    if dec.wo_recycle == 1
        nz.mix = 7;
        nz.recComp = 7;
    end
end

% Total number of algebraic variables:
n.z = nz.makeupComp + nz.mix +nz.heater + nz.split+nz.preheat+nz.mix1+nz.reactor1+nz.mix2+nz.reactor2+nz.cooler+nz.separator+nz.recComp+nz.alpha;
% Define algebraic variables:
z = MX.sym('z',n.z);
% Total number of differential state variables:
n.x = nx.preheat + nx.reactor1 + nx.reactor2 + nx.separator;
% Define differential state variables:
x = MX.sym('x',n.x);
% Rearrangement of makeup gas compressor variables:
z_makeupComp = z(1:nz.makeupComp);
variables{ind.makeupComp} = z_makeupComp;
nz.n = nz.makeupComp;
% Rearrangement of mix variables (recycle and make-up gas)
z_mix = z(nz.n+1:nz.n+nz.mix);
variables{ind.mix} = z_mix;
nz.n = nz.n+nz.mix;
% Rearrangement of heater variables
z_heater = z(nz.n+1:nz.n+nz.heater);
variables{ind.heater} = z_heater;
nz.n = nz.n+nz.heater;
% Rearrangement of split variables
z_split = z(nz.n+1:nz.n+nz.split);
variables{ind.split} = z_split;
nz.n = nz.n+nz.split;
% Rearrangement of preheat variables
z_preheat = z(nz.n+1:nz.n+nz.preheat);
x_preheat = x(nx.reactor1+nx.reactor2+1);
variables{ind.preheat} = [z_preheat(1:2); x_preheat; z_preheat(3:20)];
nz.n = nz.n + nz.preheat;
% Rearrangement of mixing 1 variables
z_mix1 = z(nz.n+1:nz.n+nz.mix1);
variables{ind.mix1} = z_mix1;
nz.n = nz.n +nz.mix1;
% Rearrangement of reactor bed 1 variables
z_reactor1 = z(nz.n+1:nz.n+nz.reactor1);
z_reactor1 = reshape(z_reactor1,6,nx.reactor1);
x_reactor1 = x(1:nx.reactor1);
a_reactor1 = [z_reactor1(1:2,:); x_reactor1'; z_reactor1(3:6,:)];
variables{ind.reactor1} = reshape(a_reactor1,nz.reactor1+nx.reactor1,1);
nz.n = nz.n + nz.reactor1;
nx.n = nx.reactor1;
% Rearrangement of mixing 2 variables
z_mix2 = z(nz.n+1:nz.n+nz.mix2);
variables{ind.mix2} = z_mix2;
nz.n = nz.n + nz.mix2;
% Rearrangement of reactor bed 2 variables
z_reactor2 = z(nz.n+1:nz.n+nz.reactor2);
z_reactor2 = reshape(z_reactor2,6,nx.reactor2);
x_reactor2 = x(nx.n+1:nx.n+nx.reactor2);
a_reactor2 = [z_reactor2(1:2,:); x_reactor2'; z_reactor2(3:6,:)];
variables{ind.reactor2} = reshape(a_reactor2,nz.reactor2+nx.reactor2,1);
nz.n = nz.n + nz.reactor2;
nx.n = nx.n + nx.reactor2+nx.preheat;
% Rearrangement of cooler variables
z_cooler = z(nz.n+1:nz.n+nz.cooler);
variables{ind.cooler} = z_cooler;
nz.n = nz.n + nz.cooler;
% Rearrangement of separator variables
z_separator = z(nz.n+1:nz.n+nz.separator);
x_separator = x(nx.n+1:nx.n+nx.separator);
if dec.reactor == 1
    variables{ind.separator} = [];
elseif dec.dynamicPressure == 1
    variables{ind.separator} = [z_separator(1);x_separator;z_separator(2:end)];
else
    variables{ind.separator} = z_separator;
end
nz.n = nz.n + nz.separator;
nx.n = nx.n + nx.separator;
alpha = z(nz.n+1:nz.n+nz.alpha);
nz.n = nz.n + nz.alpha;
% Rearrangement of recycle compressor variables
z_recComp = z(nz.n+1:nz.n+nz.recComp);
variables{ind.recComp} = z_recComp;
nz.n = nz.n + nz.recComp;
% Definition of feed disturbances
d_f = MX.sym('d_f',7);
% Definition of disturbances in unit operations
d = cell(def.sizeMat);
d_Pressure = MX.sym('d_Pressure',1);% Reactor pressure disturbance[bar]
d{ind.mix}.d_P = d_Pressure;
d_Wmake = MX.sym('d_Wmake',1); % Makeup compressor duty disturbance[W]
d{ind.makeupComp}.d_W= d_Wmake;
d_Wrec = MX.sym('d_Wrec',1); % Recycle compressor duty disturbance[W]
d{ind.recComp}.d_W= d_Wrec;
d_u1 = MX.sym('d_u1',1); % Split factor u1 disturbance [-]
d{ind.split}.d_u1 = d_u1;
d_u2 = MX.sym('d_u2',1); % Split factor u2 disturbance [-]
d{ind.split}.d_u2 = d_u2;
d_Q1 = MX.sym('d_Q1',1); % Heater duty disturbance[W]
d{ind.heater}.d_Q1 = d_Q1;
d_Q2 = MX.sym('d_Q2',1); % Cooler duty disturbance [W]
d{ind.cooler}.d_Q2 = d_Q2;
d_Kvlv = MX.sym('d_Kvlv',1); % Purge valve opening disturbance [%]
d{ind.separator}.d_Kvlv = d_Kvlv;
% Input: feed + feed disturbance
in = f+d_f;
% Independent parameters
p = [d_f; d{ind.mix}.d_P; d{ind.makeupComp}.d_W ; d{ind.recComp}.d_W; d{ind.split}.d_u1; d{ind.split}.d_u2; d{ind.heater}.d_Q1; d{ind.cooler}.d_Q2; d{ind.separator}.d_Kvlv];
% Load the model
dec.startVanHeerden = false;
[alg,ode] = ammoniaLoop(in,r,variables,ind,def,par,itg,d,dec,0,0, alpha);
% Differential equations
ode_i = vertcat(ode{:});
% Algebraic equations
alg_i = vertcat(alg{:});
% Initial values
[x0,z0]=initAmmoniaLoop(dec, f, r, par, ind, itg);
w0 = [x0; z0];

% Solve at steady-state without disturbances:
g = [ode_i;alg_i];
g_root = substitute(g,p,0); %Set the independent parameters to zero
w_root = [x; z];
g_fun = Function('g_fun',{w_root},{g_root});
G = rootfinder('G','newton',g_fun);
[root] = full(G(w0));

root0.x = root(1:size(x,1));
root0.z = root(size(x,1)+1:end);
% Rearrangement steady-state solution
i = 0;
if dec.reactor == false
    root0.dynamicPressure = root0.x(end);
    root0.makeupComp = root0.z(i+1:nz.makeupComp); i = nz.makeupComp;
    if dec.wo_recycle == true
        root0.mix = root0.z(i+1:i+nz.mix); i = i+nz.mix;
    else
        root0.mix = root0.z(i+1:i+nz.mix); i = i+nz.mix;
    end
    root0.heater = root0.z(i+1:i+nz.heater); i = i+nz.heater;
end
root0.temperature = root0.x(1:nx.reactor1+nx.reactor2+nx.preheat);
root0.split = root0.z(i+1:i+nz.split); i = i+nz.split;
root0.preheat = root0.z(i+1:i+nz.preheat); i = i+nz.preheat;
root0.mix1 = root0.z(i+1:i+nz.mix1); i = i+nz.mix1;
root0.reactor1 = root0.z(i+1:i+nz.reactor1); i = i+nz.reactor1;
root0.mix2 = root0.z(i+1:i+nz.mix2); i = i+nz.mix2;
root0.reactor2 = root0.z(i+1:i+nz.reactor2); i = i+nz.reactor2;
if dec.reactor == false
    root0.cooler = root0.z(i+1:i+nz.cooler); i = i+nz.cooler;
    if dec.dynamicPressure == 1
        root0.separator = root0.z(i+1:i+nz.separator); i = i + nz.separator;
    else
        root0.separator = root0.z(i+1:i+nz.separator); i = i + nz.separator;
    end
    root0.alpha = root0.z(i+1:i+nz.alpha); i = i + nz.alpha;
    if dec.wo_recycle == true
        root0.recComp = root0.z(i+1:i+nz.recComp); i = i+nz.recComp;
        % Calculate conversion
        root0.conv = (root0.mix(1)*root0.mix(4)-root0.preheat(1)*root0.preheat(3))/(root0.mix(1)*root0.mix(4));
        root0.conv1 = (root0.mix1(1)*root0.mix1(5)-root0.reactor1(par{ind.reactor1}.ns*6-5)*root0.reactor1(par{ind.reactor1}.ns*6-2))/(root0.mix1(1)*root0.mix1(5));
        root0.conv2 = (root0.mix2(1)*root0.mix2(5)-root0.preheat(1)*root0.preheat(4))/(root0.mix2(1)*root0.mix2(5));
    else
        root0.conv = (root0.mix(8)*root0.mix(12)-root0.preheat(1)*root0.preheat(4))/(root0.mix(8)*root0.mix(12));
        root0.conv1 = (root0.mix1(1)*root0.mix1(5)-root0.reactor1(par{ind.reactor1}.ns*6-5)*root0.reactor1(par{ind.reactor1}.ns*6-2))/(root0.mix1(1)*root0.mix1(5));
        root0.conv2 = (root0.mix2(1)*root0.mix2(5)-root0.preheat(1)*root0.preheat(4))/(root0.mix2(1)*root0.mix2(5));
    end
end
if dec.reactor == true
    % Calculate conversion
    root0.conv = (f(1)*f(5)-root0.preheat(1)*root0.preheat(4))/(f(1)*f(5));
    root0.conv1 = (f(1)*f(5)-root0.reactor1(par{ind.reactor1}.ns*6-5)*root0.reactor1(par{ind.reactor1}.ns*6-2))/(f(1)*f(5));
    root0.conv2 = (root0.mix2(1)*root0.mix2(5)-root0.preheat(1)*root0.preheat(4))/(root0.mix2(1)*root0.mix2(5));
end
%% Integration
if dec.dynamic == 1
    % Define time variables:
    t0 = 1; % start [s]
    ts = 1; % time step [s]
    tf = 5*60*60; % final [s]
    tsamp = (t0:ts:tf)/ts;
    N = length(tsamp);
    % Predefinition of solution vectors
    solution.x = zeros(length(root0.x),N+1);
    solution.x(:,1) = root0.x;
    solution.z = zeros(length(root0.z),N+1);
    solution.z(:,1) = root0.z;
    solution.temperature = zeros(length(root0.temperature),N+1);
    solution.temperature(:,1) = root0.temperature;
    if dec.dynamicPressure == true && dec.reactor == false
        solution.dynamicPressure = zeros(length(root0.dynamicPressure),N+1);
        solution.dynamicPressure(:,1) = root0.dynamicPressure;
    end
    if dec.reactor == false
        solution.makeupComp = zeros(length(root0.makeupComp),N+1);
        solution.makeupComp(:,1) = root0.makeupComp;
        solution.mix = zeros(length(root0.mix),N+1);
        solution.mix(:,1) = root0.mix;
        solution.heater = zeros(length(root0.heater),N+1);
        solution.heater(:,1) = root0.heater;
    end
    solution.split = zeros(length(root0.split),N+1);
    solution.split(:,1) = root0.split;
    solution.preheat = zeros(length(root0.preheat),N+1);
    solution.preheat(:,1) = root0.preheat;
    solution.mix1 = zeros(length(root0.mix1),N+1);
    solution.mix1(:,1) = root0.mix1;
    solution.reactor1 = zeros(length(root0.reactor1),N+1);
    solution.reactor1(:,1) = root0.reactor1;
    solution.mix2 = zeros(length(root0.mix2),N+1);
    solution.mix2(:,1) = root0.mix2;
    solution.reactor2 = zeros(length(root0.reactor2),N+1);
    solution.reactor2(:,1) = root0.reactor2;
    if dec.reactor == false
        solution.cooler = zeros(length(root0.cooler),N+1);
        solution.cooler(:,1) = root0.cooler;
        solution.separator= zeros(length(root0.separator),N+1);
        solution.separator(:,1) = root0.separator;
        solution.alpha = zeros(1,N+1);
        solution.alpha(1) = root0.alpha;
    end
    solution.conv = zeros(1,N+1);
    solution.conv(1) = root0.conv;
    solution.conv1 = zeros(1,N+1);
    solution.conv1(1) = root0.conv1;
    solution.conv2 = zeros(1,N+1);
    solution.conv2(1) = root0.conv2;
    % PI flow controller FC settings:
    tuning.Kc_f = 1000000;
    tuning.tau_I_f = 1;
    tuning.ys_f = root0.makeupComp(1)*ones(1,N+1);
    tuning.e_f = zeros(1,N+1);
    delta_u_f = zeros(1,N+1);
    % PI temperature controller TC2 settings:
    % Control inlet temperature to 1st bed
    tuning.Kc_T = -1.3800e-04;
    tuning.tau_I_T = 0.1;
    tuning.ys_T = root0.mix1(3)*ones(1,N+1);
    tuning.e_T = zeros(1,N+1);
    delta_u_T = zeros(1,N+1);
    % I-only Valve position controller (VPC)
    tuning.Ki_VPC = 10000;
    tuning.e_VPC = zeros(1,N+1);
    delta_u_VPC = zeros(1,N+1);
    % PI temperature controller TC3 settings: heater outlet temperature
    tuning.Kc_T3 = 837;
    tuning.tau_I_T3 = 1;
    tuning.ys_T3 = root0.heater(3)*ones(1,N+1);
    tuning.e_T3 = zeros(1,N+1);
    delta_u_T3 = zeros(1,N+1);
    % PI pressure controller PC settings:
    tuning.Kc_P = -1.4255e+03;
    tuning.tau_I_P = 396;
    tuning.ys_P = root0.mix(2)*ones(1,N+1); %Pressure at mixing point setpoint
    tuning.e_P = zeros(1,N+1);
    delta_u_P = zeros(1,N+1);
    % Define DAE system
    dae = struct('x',x,'z',z,'p',p,'ode',ode_i,'alg',alg_i);
    % Define integrator
    opts = struct('tf',ts); %integrating with time step ts
    func = integrator('func','idas',dae,opts);
    % Define disturbance in feed stream
    p0 = zeros(15,N+1);

    tic
    for j = 2:N+1
        res = func('x0', solution.x(:,j-1), 'z0', solution.z(:,j-1),'p', p0(:,j-1));
        solution.x(:,j) = full(res.xf);
        solution.z(:,j) = full(res.zf);
        % Rearrangement of solution:
        i = 0;
        solution.temperature(:,j) = solution.x(1:nx.reactor1+nx.reactor2+nx.preheat,j);
        if dec.reactor == false
            solution.dynamicPressure(:,j) = solution.x(end,j);
            solution.makeupComp(:,j) = solution.z(i+1:nz.makeupComp,j); i = nz.makeupComp;
            if dec.wo_recycle == true
                solution.mix(:,j) = solution.z(i+1:i+nz.mix,j); i = i+nz.mix;
            else
                solution.mix(:,j) = solution.z(i+1:i+nz.mix,j); i = i+nz.mix;
            end
            solution.heater(:,j) = solution.z(i+1:i+nz.heater,j); i = i+nz.heater;
        end
        solution.split(:,j) = solution.z(i+1:i+nz.split,j); i = i+nz.split;
        solution.preheat(:,j) = solution.z(i+1:i+nz.preheat,j); i = i+nz.preheat;
        solution.mix1(:,j) = solution.z(i+1:i+nz.mix1,j); i = i+nz.mix1;
        solution.reactor1(:,j) = solution.z(i+1:i+nz.reactor1,j); i = i+nz.reactor1;
        solution.mix2(:,j) = solution.z(i+1:i+nz.mix2,j); i = i+nz.mix2;
        solution.reactor2(:,j) = solution.z(i+1:i+nz.reactor2,j); i = i+nz.reactor2;
        if dec.reactor == false
            solution.cooler(:,j) = solution.z(i+1:i+nz.cooler,j); i = i+nz.cooler;
            solution.conv(j) = (solution.mix(8,j)*solution.mix(12,j)-solution.preheat(1,j)*solution.preheat(4,j))/(solution.mix(8,j).*solution.mix(12,j));
            solution.conv1(j) = (solution.mix1(1,j)*solution.mix1(5,j)-solution.reactor1(par{ind.reactor1}.ns*6-5,j)*solution.reactor1(par{ind.reactor1}.ns*6-2,j))/(solution.mix1(1,j)*solution.mix1(5,j));
            solution.conv2(j) = (solution.mix2(1,j)*solution.mix2(5,j)-solution.preheat(1,j)*solution.preheat(4,j))/(solution.mix2(1,j)*solution.mix2(5,j));
            if dec.dynamicPressure == 1
                solution.separator(:,j) = solution.z(i+1:i+nz.separator,j);
                i = i + nz.separator;
            else
                solution.separator(:,j) = solution.z(i+1:i+nz.separator,j);
                i = i + nz.separator;
            end
            if dec.wo_recycle ==1
                solution.recComp(:,j) = solution.z(i+1:i+nz.recComp,j);
            end
        end
        if dec.makeupController == true
            % PI flow controller FC
            tuning.e_f(j) = tuning.ys_f(j) - solution.makeupComp(1,j);
            delta_u_f(j) = PID(tuning.Kc_f, tuning.tau_I_f,ts,tuning.e_f(j),tuning.e_f(j-1));
        end
        p0(9,j+1) = p0(9,j) + delta_u_f(j);
        if dec.tempController1 == true
            % PI temperature controller TC2
            tuning.e_T(j) = tuning.ys_T(j) - solution.mix1(3,j);
            delta_u_T(j) = PID(tuning.Kc_T, tuning.tau_I_T,ts,tuning.e_T(j),tuning.e_T(j-1));
        end
        p0(11,j+1) = p0(11,j) + delta_u_T(j);
        % Constraints
        u = par{ind.split}.u1+p0(11,j+1)+par{ind.split}.u2+p0(12,j+1);
        u1 = par{ind.split}.u1+p0(11,j+1);
        u2 = par{ind.split}.u2+p0(12,j+1);
        if (u>1 || u1<0 || u2<0)
            p0(11,j+1) = p0(11,j);
            p0(12,j+1) = p0(12,j);
        end
        % Valve position control VPC
        if (u1<0.15) && dec.VPC == true
            tuning.e_VPC(j) = 0.15-u1;
            delta_u_VPC(j) = I_controller(tuning.Ki_VPC,tuning.e_VPC(j));
            p0(13,j+1) = p0(13,j) + delta_u_VPC(j);
        end
        if dec.tempController3 == true
            % PI temperature controller TC3
            tuning.e_T3(j) = tuning.ys_T3(j) - solution.heater(3,j);
            delta_u_T3(j) = PID(tuning.Kc_T3, tuning.tau_I_T3,ts,tuning.e_T3(j),tuning.e_T3(j-1));
            p0(13,j+1) = p0(13,j) + delta_u_T3(j);
        end
        if dec.pressureController == true
            % PI pressure controller PC
            tuning.e_P(j) = tuning.ys_P(j) - solution.mix(2,j);
            delta_u_P(j) = PID(tuning.Kc_P, tuning.tau_I_P,ts,tuning.e_P(j),tuning.e_P(j-1));
            p0(10,j+1) = p0(10,j) + delta_u_P(j);
        end
    end
    toc
end
%% Van Heerden Analysis
if (dec.vanHeerden == true && dec.reactor == true)
    dec.startVanHeerden = true;
    % Independent reactor inlet temperature
    Ti = MX.sym('Ti',1);
    % Independent reactor outlet temperature
    To = MX.sym('To',1);
    %Load the model
    [alg,ode] = ammoniaLoop(f,r,variables,ind,def,par,scl,itg,d,dec,Ti,To);
    %Differential equations
    ode_temp = vertcat(ode{:});
    %Algebraic constraints
    alg_temp = vertcat(alg{:});
    % Define inlet and outlet temperature range
    Ti_list = (50:1:600)./scl_T1;
    To_list = Ti_list;
    nsamp = length(Ti_list);
    % Initial values
    [x0, z0] = initVanHeerden(f, itg);
    w0_temp = [x0; z0];
    % Define solver
    g = [ode_temp;alg_temp];
    w_root = [x;z];
    g_temp = substitute(g, p, 0);
    g_temp1 = substitute(g_temp, Ti, Ti_list(1));
    g_temp1 = substitute(g_temp1, To, To_list(1));
    g_fun = Function('g_fun',{w_root},{g_temp1});
    G = rootfinder('G','newton',g_fun);
    % Predefine solution vector
    root_temp = zeros(length(w_root),nsamp);
    Treactor_out = zeros(1,nsamp);
    Treactor_in = zeros(1,nsamp);
    % Solve at first time step
    root_temp(:,1) = full(G(w0_temp));
    Treactor_out(1) = root_temp(nx.reactor1+nx.reactor2+nx.preheat,1).*scl_T1; %reactor outlet temperature
    Treactor_in(1) = root_temp(n.x+nz.split+6+7+3,1).*scl_T1;
    % Solve for remaining time steps
    for j = 2:nsamp
        g_temp2 = substitute(g_temp, Ti, Ti_list(j));
        g_temp2 = substitute(g_temp2, To, To_list(j));
        g_fun = Function('g_fun',{w_root},{g_temp2});
        G = rootfinder('G','newton',g_fun);
        root_temp(:,j) = full(G(root_temp(:,j-1)));
        Treactor_out(j) = root_temp(nx.reactor1+nx.reactor2+nx.preheat,j).*scl_T1;
        Treactor_in(j) = root_temp(n.x+nz.split+6+7+3,j).*scl_T1;
    end
end

