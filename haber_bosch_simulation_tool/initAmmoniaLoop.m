function [x0, z0] = initAmmoniaLoop(dec, f, r, par, ind, itg)
% This function is an educated guess made by Clause AI as the original init
% file is not available. The main code still does not work, but it is
% unclear whether a working init file is the missing piece.

% INITAMMONIALOOP  Initial guess vectors for the ammonia loop DAE system.
%
%   [x0, z0] = initAmmoniaLoop(dec, f, r, par, ind, itg)
%
%   Returns initial guesses for the differential state vector x0 and the
%   algebraic variable vector z0.  All block sizes are derived from the
%   input arguments so the function stays consistent with parAmmoniaLoop.m
%   automatically.  Every layout has been verified against ammoniaLoop.m.
%
% -------------------------------------------------------------------------
%  INPUTS
%    dec  - decision flags:
%             dec.reactor          true = reactor section only
%             dec.wo_recycle       true = no material recycle
%             dec.dynamicPressure  true = separator pressure is differential
%    f    - make-up feed stream vector (7 x 1)
%             [F(kmol/s), P(bar), T(C), yH2, yN2, yNH3, yI]
%    r    - recycle stream vector (7 x 1), same convention as f
%    par  - cell array of unit-operation parameters (from parAmmoniaLoop.m)
%    ind  - struct of unit-operation indices (from parAmmoniaLoop.m)
%    itg  - struct of integer values:
%             itg.k = 7  (variables per stream)
%             itg.s = 4  (species)
%
% -------------------------------------------------------------------------
%  DIFFERENTIAL STATE VECTOR  x0
%
%    x(1 : ns1)              bed-1 node temperatures [C]
%    x(ns1+1 : ns1+ns2-1)    bed-2 node temperatures [C]
%    x(ns1+ns2)              preheat hot-inlet temperature [C]
%                            = Tin_h = reactor bed-2 outlet T;
%                              carries the thermal inertia of HX2
%    x(ns1+ns2+1)            separator pressure [bar]
%                            (only when dec.dynamicPressure == true)
%
% -------------------------------------------------------------------------
%  ALGEBRAIC VARIABLE VECTOR  z0   (blocks in main-script order)
%
%  Full loop only (dec.reactor == false):
%
%   makeupComp (8)
%     Verified: into{C1} = [v{C1}(1); f(2:7); v{C1}(2:8)]
%     z(1)    = F_in   make-up inlet flow (P,T,comp taken directly from f)
%     z(2:8)  = compressor outlet stream [F,P,T,yH2,yN2,yNH3,yI]
%
%   mix  (14  or  7 if wo_recycle)
%     Verified: into{mix} = [v{C1}(2:8); v{mix}(1:7); v{mix}(8:14)]  (normal)
%               into{mix} = [v{C1}(2:8); r;           v{mix}(1:7) ]  (wo_recycle)
%     Normal    z(1:7)  = recycle stream,  z(8:14) = mixed outlet
%     wo_recycle z(1:7) = mixed outlet only
%
%   heater  (7)
%     Verified: into{HX1} = [into{mix}(k2+1:k3); v{HX1}(1:7)]
%     z(1:7) = heater outlet stream
%
%  Both modes:
%
%   split  (21)
%     Verified: into{split} = [inlet(7); v{split}(1:7); v{split}(8:14); v{split}(15:21)]
%     z(1:7)   = quench-1 outlet  -> to mix1
%     z(8:14)  = quench-2 outlet  -> to mix2
%     z(15:21) = main-feed outlet -> to preheat cold inlet
%
%   preheat  (20)
%     Verified: into{HX2} = [v{pre}(1:7); v{pre}(8:14);
%                             v{split}(15:21); v{pre}(15:21)]
%     with v{pre} = [z_pre(1:2); x_pre; z_pre(3:20)]
%     Cold inlet NOT stored here -- read from v{split}(15:21) in ammoniaLoop.m.
%     z(1)     = nin_h   hot inlet flow      [kmol/s]
%     z(2)     = Pin_h   hot inlet pressure  [bar]
%     -- x_preheat = Tin_h  hot inlet T (DIFFERENTIAL, not in z) --
%     z(3:6)   = xin_h   hot inlet composition [yH2,yN2,yNH3,yI]
%     z(7)     = nout_h  hot outlet flow     [kmol/s]
%     z(8)     = Pout_h  hot outlet pressure [bar]
%     z(9)     = Tout_h  hot outlet temp     [C]
%     z(10:13) = xout_h  hot outlet composition
%     z(14)    = nout_c  cold outlet flow    [kmol/s]
%     z(15)    = Pout_c  cold outlet pressure[bar]
%     z(16)    = Tout_c  cold outlet temp    [C]
%     z(17:20) = xout_c  cold outlet composition
%
%   mix1  (7)
%     z(1:7) = bed-1 inlet stream (blend of preheat cold outlet + quench-1)
%
%   reactor1  (ns1 * (itg.k-1))
%     Per node: [F, P, yH2, yN2, yNH3, yI]  (T is in x0)
%
%   mix2  (7)
%     z(1:7) = bed-2 inlet stream (blend of bed-1 outlet + quench-2)
%
%   reactor2  ((ns2-1) * (itg.k-1))
%     Per node: [F, P, yH2, yN2, yNH3, yI]  (T is in x0)
%
%  Full loop only (dec.reactor == false):
%
%   cooler  (7)
%     z(1:7) = cooler outlet stream
%
%   separator  (21  or  20 if dynamicPressure)
%     Without dynamicPressure:
%       z(1:7)   = product outlet (liquid, NH3-rich)
%       z(8:14)  = purge outlet
%       z(15:21) = recycle outlet (to compressor C2)
%     With dynamicPressure -- x_sep = P_sep inserted at position 2:
%       variables{sep} = [z_sep(1); x_sep; z_sep(2:20)]
%       z(1)    = F_liq   product flow
%       -- x_sep = P_sep  separator pressure (DIFFERENTIAL) --
%       z(2:6)  = product(3:7) = [T_liq, yH2, yN2, yNH3, yI]
%       z(7:13) = purge outlet (7)
%       z(14:20)= recycle outlet (7)
%
%   alpha  (1)   vapour recycle split fraction
%
%   recComp  (0  or  7 if wo_recycle)
%     Normal:     outlet = v{mix}(1:7) already in z_mix, so nz = 0
%     wo_recycle: z(1:7) = compressor outlet stream
%
% =========================================================================

% ---- Sizing from arguments (no hardcoding) ------------------------------
ns1   = par{ind.reactor1}.ns;       % e.g. 20
ns2   = par{ind.reactor2}.ns;       % e.g. 20
k     = itg.k;                      % 7
k_alg = k - 1;                      % 6  algebraic vars per reactor node
nx_r1 = ns1;                        % differential temps in bed 1
nx_r2 = ns2 - 1;                    % differential temps in bed 2

% ---- Split fractions (from par) -----------------------------------------
u1 = par{ind.split}.u1;             % e.g. 0.2302
u2 = par{ind.split}.u2;             % e.g. 0.2659

% ---- Parameter values used in initial estimates -------------------------
W_makeup = par{ind.makeupComp}.Wcomp;   % compressor duty [W]
W_rec    = par{ind.recComp}.Wcomp;      % recycle compressor duty [W]

% =========================================================================
%  DERIVE NOMINAL STREAM VALUES FROM f AND r
%  All stream vectors: [F(kmol/s), P(bar), T(C), yH2, yN2, yNH3, yI]
% =========================================================================

% -- Make-up compressor outlet: same composition as f, compressed to r(2) --
makeup_out      = f;
makeup_out(1)   = max(f(1), 0.0141);  % use nominal flow if feed specifies 0
makeup_out(2)   = r(2);               % loop pressure
makeup_out(3)   = f(3) + 65;          % ~65 C temperature rise from compression

% -- Mixed stream (make-up + recycle) -------------------------------------
if dec.wo_recycle
    % No recycle: mix outlet = makeup outlet scaled to reactor conditions
    mix_out   = makeup_out;
    mix_out(2) = r(2);
    mix_out(3) = f(3) + 10;
else
    % Blend make-up outlet and recycle stream by flow-weighted average
    F_mk  = makeup_out(1);
    F_rec = r(1);
    F_mix = F_mk + F_rec;
    mix_out    = zeros(7,1);
    mix_out(1) = F_mix;
    mix_out(2) = r(2);                          % pressure set by recycle
    mix_out(3) = (F_mk*makeup_out(3) + F_rec*r(3)) / F_mix;
    mix_out(4) = (F_mk*makeup_out(4) + F_rec*r(4)) / F_mix;  % yH2
    mix_out(5) = (F_mk*makeup_out(5) + F_rec*r(5)) / F_mix;  % yN2
    mix_out(6) = (F_mk*makeup_out(6) + F_rec*r(6)) / F_mix;  % yNH3
    mix_out(7) = (F_mk*makeup_out(7) + F_rec*r(7)) / F_mix;  % yI
end

% -- Heater outlet: same flow/comp, temperature raised by HX1 duty --------
heater_out    = mix_out;
dT_heater     = par{ind.heater}.Q / (mix_out(1) * par{ind.heater}.Cp);
heater_out(3) = mix_out(3) + dT_heater;

% -- Reactor inlet (in reactor-only mode, f IS the reactor inlet) ---------
if dec.reactor
    reactor_inlet = f;
else
    reactor_inlet = heater_out;
end

% -- Split outlets --------------------------------------------------------
F_tot    = reactor_inlet(1);
quench1  = reactor_inlet; quench1(1)   = F_tot * u1;
quench2  = reactor_inlet; quench2(1)   = F_tot * u2;
main_feed= reactor_inlet; main_feed(1) = F_tot * (1 - u1 - u2);

% -- Approximate reactor outlet (used for preheat hot side and bed profiles)
% Estimate: NH3 mole fraction increases by ~0.15, H2/N2 decrease by stoich
% N2 + 3H2 -> 2NH3, stoi = [-3 -1 2 0]
dNH3_approx = 0.15;
reactor_out      = reactor_inlet;
reactor_out(4)   = max(reactor_inlet(4) - 1.5*dNH3_approx, 0.05);  % yH2
reactor_out(5)   = max(reactor_inlet(5) - 0.5*dNH3_approx, 0.05);  % yN2
reactor_out(6)   = min(reactor_inlet(6) + dNH3_approx,      0.30);  % yNH3
reactor_out(7)   = reactor_inlet(7);                                 % yI
% renormalise compositions
y_sum            = sum(reactor_out(4:7));
reactor_out(4:7) = reactor_out(4:7) / y_sum;
reactor_out(2)   = reactor_inlet(2) - 2.0;   % ~2 bar pressure drop
reactor_out(3)   = 460.0;                     % typical bed outlet [C]

% -- Cooler outlet: same flow/comp as reactor out, cooled to separator T --
T_sep_nom  = 12.5;    % [C] nominal separator temperature
cooler_out = reactor_out;
cooler_out(3) = T_sep_nom;
cooler_out(2) = reactor_out(2) - 0.5;

% -- Separator streams ----------------------------------------------------
% Liquid product: NH3-rich, most NH3 condensed
sep_product    = cooler_out;
sep_product(1) = cooler_out(1) * cooler_out(6) * 0.90;  % ~90% NH3 recovered
sep_product(3) = T_sep_nom;
sep_product(4) = 0.010;   % yH2  (trace)
sep_product(5) = 0.005;   % yN2  (trace)
sep_product(6) = 0.970;   % yNH3 (concentrated)
sep_product(7) = 0.015;   % yI   (trace)

% Recycle vapour: lean in NH3
F_liq          = sep_product(1);
F_vap          = cooler_out(1) - F_liq;
sep_recycle    = cooler_out;
sep_recycle(1) = F_vap * 0.95;    % 95% of vapour goes to recycle (alpha)
sep_recycle(3) = T_sep_nom;
sep_recycle(4) = r(4);  % use given recycle compositions as best estimate
sep_recycle(5) = r(5);
sep_recycle(6) = r(6);
sep_recycle(7) = r(7);

% Purge: small fraction of vapour
sep_purge      = sep_recycle;
sep_purge(1)   = F_vap * 0.05;

% =========================================================================
%  DIFFERENTIAL STATE VECTOR  x0
% =========================================================================

% Bed 1: ns1 nodes, temperature rises from reactor inlet to ~480 C
T_bed1_in  = main_feed(3) + (quench1(1)*quench1(3)) / (main_feed(1) + quench1(1));
T_bed1_in  = max(T_bed1_in, reactor_inlet(3));   % at least feed temperature
T_bed1 = linspace(T_bed1_in, 480, nx_r1)';

% Bed 2: ns2-1 nodes, re-cooled inter-bed inlet rises to ~460 C
T_bed2_in  = (reactor_out(1)*reactor_out(3) + quench2(1)*quench2(3)) / ...
             (reactor_out(1) + quench2(1));
T_bed2 = linspace(T_bed2_in, 460, nx_r2)';

% Preheat hot-inlet temperature = reactor bed-2 outlet temperature
Tin_h_ss = reactor_out(3);   % [C]

x0 = [T_bed1; T_bed2; Tin_h_ss];

if ~dec.reactor && dec.dynamicPressure
    x0 = [x0; cooler_out(2)];   % separator pressure [bar]
end

% =========================================================================
%  ALGEBRAIC VARIABLE VECTOR  z0
% =========================================================================
z0 = [];

% ------------------------------------------------------------------
%  Header blocks  (full-loop mode only)
% ------------------------------------------------------------------
if ~dec.reactor

    % makeupComp  (8): [F_in(1), outlet_stream(7)]
    z0 = [z0; f(1); makeup_out];

    % mix  (14 or 7)
    if dec.wo_recycle
        z0 = [z0; mix_out];
    else
        z0 = [z0; r; mix_out];     % [recycle_stream(7), mixed_outlet(7)]
    end

    % heater  (7): outlet stream only
    z0 = [z0; heater_out];

end

% ------------------------------------------------------------------
%  Split  (21)
% ------------------------------------------------------------------
z0 = [z0; quench1; quench2; main_feed];

% ------------------------------------------------------------------
%  Preheat HX2  (20)
% ------------------------------------------------------------------
% Hot side = reactor outlet (bed-2 outlet feeds into HX2 shell/tube)
% Cold side inlet = main_feed (from split) -- NOT stored in z_preheat
nin_h  = reactor_out(1);
Pin_h  = reactor_out(2);
xin_h  = reactor_out(4:7);
nout_h = reactor_out(1);           % flow conserved
Pout_h = reactor_out(2) - 0.3;    % small pressure drop
Tout_h = heater_out(3) - 10;      % hot side exits just above heater inlet T
xout_h = reactor_out(4:7);        % composition unchanged in HX
nout_c = main_feed(1);            % cold outlet flow = cold inlet flow
Pout_c = main_feed(2) - 0.3;     % small pressure drop
Tout_c = heater_out(3);           % cold side exits ~= heater outlet T (preheated)
xout_c = main_feed(4:7);          % composition unchanged in HX

z_preheat = [nin_h; Pin_h; xin_h; nout_h; Pout_h; Tout_h; xout_h; ...
             nout_c; Pout_c; Tout_c; xout_c];   % 2+4+7+7 = 20
z0 = [z0; z_preheat];

% ------------------------------------------------------------------
%  Mix1  (7): preheat cold outlet + quench1 -> bed-1 inlet
% ------------------------------------------------------------------
F_mix1     = nout_c + quench1(1);
T_mix1     = (nout_c * Tout_c + quench1(1) * quench1(3)) / F_mix1;
mix1_out   = main_feed;
mix1_out(1)= F_mix1;
mix1_out(3)= T_mix1;
z0 = [z0; mix1_out];

% ------------------------------------------------------------------
%  Reactor bed 1  (ns1 * k_alg variables)
% ------------------------------------------------------------------
yH2_1  = linspace(mix1_out(4),  reactor_out(4)+0.05, ns1);
yN2_1  = linspace(mix1_out(5),  reactor_out(5)+0.02, ns1);
yNH3_1 = linspace(mix1_out(6),  reactor_out(6)*0.6,  ns1);
yI_1   = linspace(mix1_out(7),  reactor_out(7),       ns1);
% Renormalise row-wise
Y1 = [yH2_1; yN2_1; yNH3_1; yI_1];
Y1 = Y1 ./ sum(Y1, 1);
F_1 = linspace(mix1_out(1), mix1_out(1)*0.98, ns1);
P_1 = linspace(mix1_out(2), reactor_out(2)+0.5, ns1);

z_r1 = zeros(ns1 * k_alg, 1);
for i = 1:ns1
    s = (i-1)*k_alg + 1;
    z_r1(s:s+k_alg-1) = [F_1(i); P_1(i); Y1(1,i); Y1(2,i); Y1(3,i); Y1(4,i)];
end
z0 = [z0; z_r1];

% ------------------------------------------------------------------
%  Mix2  (7): bed-1 outlet + quench2 -> bed-2 inlet
% ------------------------------------------------------------------
bed1_out    = reactor_out;
bed1_out(3) = T_bed1(end);        % bed-1 outlet T = last node of bed1
F_mix2      = bed1_out(1) + quench2(1);
T_mix2      = (bed1_out(1)*bed1_out(3) + quench2(1)*quench2(3)) / F_mix2;
mix2_out    = reactor_out;
mix2_out(1) = F_mix2;
mix2_out(3) = T_mix2;
z0 = [z0; mix2_out];

% ------------------------------------------------------------------
%  Reactor bed 2  ((ns2-1) * k_alg variables)
% ------------------------------------------------------------------
yH2_2  = linspace(mix2_out(4),  reactor_out(4),  ns2-1);
yN2_2  = linspace(mix2_out(5),  reactor_out(5),  ns2-1);
yNH3_2 = linspace(mix2_out(6),  reactor_out(6),  ns2-1);
yI_2   = linspace(mix2_out(7),  reactor_out(7),  ns2-1);
Y2 = [yH2_2; yN2_2; yNH3_2; yI_2];
Y2 = Y2 ./ sum(Y2, 1);
F_2 = linspace(mix2_out(1), reactor_out(1), ns2-1);
P_2 = linspace(mix2_out(2), reactor_out(2), ns2-1);

z_r2 = zeros((ns2-1) * k_alg, 1);
for i = 1:(ns2-1)
    s = (i-1)*k_alg + 1;
    z_r2(s:s+k_alg-1) = [F_2(i); P_2(i); Y2(1,i); Y2(2,i); Y2(3,i); Y2(4,i)];
end
z0 = [z0; z_r2];

% ------------------------------------------------------------------
%  Tail blocks  (full-loop mode only)
% ------------------------------------------------------------------
if ~dec.reactor

    % cooler HX3  (7): outlet stream only
    z0 = [z0; cooler_out];

    % separator S1  (21 or 20)
    if dec.dynamicPressure
        % 20 algebraic: [F_liq(1), product(3:7)(5), purge(7), recycle(7)]
        % x_sep = P_sep inserted at position 2 of full block by main script
        z_sep = [sep_product(1); sep_product(3:7); sep_purge; sep_recycle];
    else
        % 21 algebraic: three full outlet streams
        z_sep = [sep_product; sep_purge; sep_recycle];
    end
    z0 = [z0; z_sep];

    % alpha  (1): vapour recycle split fraction
    z0 = [z0; 0.95];

    % recComp C2  (0 or 7)
    if dec.wo_recycle
        z_recComp    = sep_recycle;
        z_recComp(2) = r(2);              % compress back to loop pressure
        dT_rec       = W_rec / (sep_recycle(1) * par{ind.recComp}.Cp);
        z_recComp(3) = sep_recycle(3) + dT_rec;
        z0 = [z0; z_recComp];
    end

end

% Ensure column vectors
x0 = x0(:);
z0 = z0(:);

end