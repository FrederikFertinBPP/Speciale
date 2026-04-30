% This scripts defines the parameters of the different unit operations
% Unit operations in the loop
name.ammoniaLoop = {'makeupcompressor';'mix';'heater';'split'; 'preheat';'mix1'; 'reactorbed1';'mix2'; 'reactorbed2';'cooler'; 'separator';'recyclecompessor'};

def.sizeMat = size(name.ammoniaLoop);
par = cell(def.sizeMat);
scl = cell(def.sizeMat);
% Index of each unit operations
ind.makeupComp = 1;
ind.mix = 2;
ind.heater = 3;
ind.split = 4;
ind.preheat = 5;
ind.mix1 = 6;
ind.reactor1 = 7;
ind.mix2 = 8;
ind.reactor2 = 9;
ind.cooler = 10;
ind.separator = 11;
ind.recComp = 12;
% Define intiger values
itg.k = 7; % process variables in each stream
itg.s = 4; % species in each stream
%% ********* Parameters of makeup compressor C1 ***************
par{ind.makeupComp}.R = 8314; % J/(kmol K)
par{ind.makeupComp}.Cp = 31055; % J/(kmol K)
par{ind.makeupComp}.Wcomp = 4.1748e+05; % W
par{ind.makeupComp}.nc = 0.85;
%% ********* Parameters of heater HX1 **************************
par{ind.heater}.Cp = 31055; % J/(kmol K)
par{ind.heater}.Q = 9.7651e+05; % W
%% ********* Parameters of splitter (quench flow)***************
par{ind.split}.u1 = (0.2302);
par{ind.split}.u2 = 0.1389+0.1270;
%% ************ Parameters of preheater HX2 *******************
par{ind.preheat}.U = 536; % W/(m2 K)
par{ind.preheat}.A = 50; % m2
par{ind.preheat}.Cpc = 31055; % J/(kmol K)
par{ind.preheat}.Cph = 31055; % J/(kmol K)
%% ********** Parameters of reactor bed R1 **********************
par{ind.reactor1}.ns = 20; % number of sections ns
par{ind.reactor1}.stoi = [-3 -1 2 0];
par{ind.reactor1}.Vbed = 0.7; % m3
par{ind.reactor1}.rhocat = 2200; % kg/m3
par{ind.reactor1}.Cp = 31055; % J/(kmol K)
par{ind.reactor1}.Cpcat = 1100; % J/(kg K)
par{ind.reactor1}.dHrx = 91.9836e6; % J/kmol
%kinetics:
par{ind.reactor1}.Afor = 1.79e4;
par{ind.reactor1}.Abac = 2.57e16;
par{ind.reactor1}.Eafor = 87090; % J/mol
par{ind.reactor1}.Eabac = 198464; % J/mol
par{ind.reactor1}.R = 8.314; % J/(mol K)
%% ********** Parameters of Reactor bed R2 *********************
par{ind.reactor2}.ns = 20; % number of sections ns
par{ind.reactor2}.stoi = [-3 -1 2 0];
par{ind.reactor2}.Vbed = 1.3; % m3
par{ind.reactor2}.rhocat = 2200; % kg/m3
par{ind.reactor2}.Cp = 31055; % J/(kmol K)
par{ind.reactor2}.Cpcat = 1100; % J/(kg K)
par{ind.reactor2}.dHrx = 91.9836e6; % J/kmol
%kinetics:
par{ind.reactor2}.Afor = 1.79e4;
par{ind.reactor2}.Abac = 2.57e16;
par{ind.reactor2}.Eafor = 87090; % J/mol
par{ind.reactor2}.Eabac = 198464; % J/mol
par{ind.reactor2}.R = 8.314; % J/(mol K)
%% *********** Cooler HX3 **************************************
par{ind.cooler}.Q = (-7.8388e+06)*0.96; % W
par{ind.cooler}.Cp = 31055; % J/(kmol K)
par{ind.cooler}.Hvap = 1.515e7; % J/(kmol)
%% *********** Separator S1 ************************************
par{ind.separator}.Ppurge = 10; % bar
par{ind.separator}.Kvlv = 6.7237e-04; % kmol/(s bar)
par{ind.separator}.Khx = 0.1210; % kmol/(s bar)
par{ind.separator}.Vtot = 3.4740; % m3
par{ind.separator}.R = 0.08314; % m3 bar/(kmol K)
par{ind.separator}.H1 = [-3.68607;-2.29337;-1.67010];
par{ind.separator}.H2 = [0.596736;0.5294740;0.440558]*10^4;
par{ind.separator}.H3 = [-0.642828;-0.521881;-0.482973]*10^6;
par{ind.separator}.A = -0.114397*10^3;
par{ind.separator}.B = 1.24673;
par{ind.separator}.C = -0.353366*10^(-2);
par{ind.separator}.D = -0.304684*10^(-5);
par{ind.separator}.E = 0.186446*10^(-7);
%% *********** Compressor C2 ***********************************
par{ind.recComp}.R = 8314; % J/(kmol K)
par{ind.recComp}.Cp = 31055; % J/(kmol K)
par{ind.recComp}.Wcomp = 5.6155e+04; % W
par{ind.recComp}.nc = 0.85;
