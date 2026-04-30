function [alg,ode] = ammoniaLoop(f,r,v,ind,def,par,itg,d,dec,Ti,To,alpha)
    % This function defines the flowsheet of the ammonia synthesis loop.
    %
    % The input of this function is defined as
    % f: Feed stream
    % r: Recycle stream
    % v: Structure of algebraic and differential state variables
    % ind: Index of unit operations
    % def: Definition of.
    % .sizeMat Number of unit operations
    % par: Structure of parameters used in unit operations
    % itg: Intiger values
    % .k Total number of variables per stream
    % d Structure of disturbances in unit operations
    % dec Decision variables
    % Ti Reactor inlet temperature (van Heerden analysis)
    % To Reactor outlet temperature (van Heerden analysis)
    % The structures used in this function are
    % into: Structure containing the input vector for the unit
    % operations.
    % Reassignment of solution vector
    into = cell(def.sizeMat);
    alg = cell(def.sizeMat);
    ode = cell(def.sizeMat);
    % Definition of multiples of itg.k
    k1 = itg.k;
    k2 = 2*itg.k;
    k3 = 3*itg.k;
    %% ************* Make-up gas compressor C1*******************
    k = ind.makeupComp;
    if size(v{k},1) ~= 0
        into{k} = [v{k}(1);f(2:7); % inlet: make-up stream
                    v{k}(2:8)]; % outlet: compressed make-up stream
        [alg{k},ode{k}] = compressor(into{k},par{k},itg,d{k}) ;
    end
    %% ************* Mix of recycle and makeup ******************
    k = ind.mix;
    if size(v{k},1) ~= 0
        if dec.wo_recycle == 1
            into{k} = [v{k-1}(2:8); % inlet: compressed make-up stream
                        r; % inlet: defined recycle stream
                        v{k}(1:k1)]; % outlet: reactor feed
        else
            into{k} = [v{k-1}(2:8); % inlet: compressed makeup stream
                        v{k}(1:k1) % inlet: recycle stream
                        v{k}(k1+1:k2)]; % outlet: reactor feed
        end
        [alg{k},ode{k}] = makeupMixer(into{k},itg,d{k});
    end
    %% ************* Simple HEX HX1 ******************************
    k = ind.heater;
    if size(v{k},1) ~= 0
        into{k} = [into{k-1}(k2+1:k3); % inlet
                    v{k}(1:k1)]; % outlet
        [alg{k},ode{k}] = simpleHEX(into{k},par{k},itg, d{k});
    end
    %% *************Split function *******************************
    k = ind.split;
    if size(v{k},1) ~= 0
        if dec.reactor == true
            into{k} = [f; % inlet: defined reactor inlet
                        v{k}(1:k1) % outlet: quench 1
                        v{k}(k1+1:k2) % outlet: quench 2
                        v{k}(k2+1:k3)]; % outlet: preheater cold inlet
        else
            into{k} = [v{k-1}(1:k1); % inlet: reactor inlet
                        v{k}(1:k1) % outlet: quench 1
                        v{k}(k1+1:k2) % outlet: quench 2
                        v{k}(k2+1:k3)]; % outlet: preheater cold inlet
        end
        [alg{k},ode{k}] = splitter(into{k}, par{k}, itg, d{k});
    end
    %% ************* Preheater HX2 ******************************
    k = ind.preheat;
    if size(v{k},1) ~= 0
        if dec.startVanHeerden == true
            into{k} = [v{k}(1:2);To;v{k}(4:k1); % inlet hot stream
                        v{k}(k1+1:k2); % outlet hot stream
                        v{k-1}(k2+1:k3) % inlet cold stream
                        v{k}(k2+1:k3)]; % outlet cold stream
        else
            into{k} = [v{k}(1:k1); % inlet hot stream
                        v{k}(k1+1:k2); % outlet hot stream
                        v{k-1}(k2+1:k3) % inlet cold stream
                        v{k}(k2+1:k3)]; % outlet cold stream
        end
        [alg{k},ode{k}] = hexNTU(into{k}, par{k}, itg);
    end
    %% ************* Mixing before Reactor Bed R1 ***************
    k = ind.mix1;
    if size(v{k},1) ~= 0
        if dec.startVanHeerden == true
            % inlet: preheat cold outlet
            into{k} = [v{k-1}(k2+1:k2+2);Ti;v{k-1}(k2+4:k3);
                        v{ind.split}(1:k1); % inlet: quench 1
                        v{k}(1:k1)]; % outlet: reactor 1 inlet
        elseif size(v{k},1) ~= 0
            into{k} = [v{k-1}(k2+1:k3); % inlet: preheat cold outlet
                        v{ind.split}(1:k1); % inlet: quench 1
                        v{k}(1:k1)]; % outlet: reactor 1 inlet
        end
        [alg{k},ode{k}] = mixer(into{k}, itg);
    end
    %% ************* Reactor bed R1 *****************************
    k = ind.reactor1;
    if size(v{k},1) ~= 0
        into{k} = [v{k-1}(1:k1); % inlet
                    v{k}]; % internal streams + outlet stream
        [alg{k},ode{k}] = reactorBed(into{k}, par{k}, itg);
    end
    %% ************* Mixing before Reactor Bed R2 ****************
    k = ind.mix2;
    if size(v{k},1) ~= 0
        into{k} = [v{k-1}(end-k1+1:end); % inlet: from reactor 1
                    v{ind.split}(k1+1:k2); % inlet: quench 2
                    v{k}(1:7)]; % outlet: reactor bed 2 inlet
        [alg{k},ode{k}] = mixer(into{k},itg);
    end
    %% ************* Reactor bed R2 ******************************
    k = ind.reactor2;
    if size(v{k},1) ~= 0
        into{k} = [v{k-1}(1:k1); % inlet: from mixer 3
                    v{k}; % internal flows
                    v{ind.preheat}(1:k1)]; % outlet flow: preheater hot inlet
        [alg{k},ode{k}] = reactorBed(into{k}, par{k}, itg);
    end
    %% *************** Cooler HX3 ********************************
    k = ind.cooler;
    if size(v{k},1) ~= 0
        into{k} = [v{ind.preheat}(k1+1:k2); % inlet: preheater hot outlet
                    v{k}(1:7)]; % outlet
        [alg{k},ode{k}] = cooler(into{k}, par{k}, itg,dec, alpha, d{k});
    end
    %% *************** Separator S1 *****************************
    k = ind.separator;
    if size(v{k},1) ~= 0
        into{k} = [v{k-1}(1:k1); % inlet: from cooler
                    v{k}(1:k1); % outlet: product stream
                    v{k}(k1+1:k2); % outlet: purge stream
                    v{k}(k2+1:k3); % outlet: stream to recycle compressor
        ];
        [alg{k},ode{k}] = separator(into{k}, par{k},itg,dec, alpha, d{k});
    end
    %% *************** Recycle compressor C2 ********************
    k = ind.recComp;
    if size(v{k-1},1) ~= 0
        if dec.wo_recycle == 1
            into{k} = [v{k-1}(k2+1:k3); % inlet
                            v{k}(1:k1)]; % outlet
        elseif size(v{ind.mix},1) ~= 0
            into{k} = [v{k-1}(k2+1:k3); % inlet
                        v{ind.mix}(1:7)]; % outlet
        end
        [alg{k},ode{k}] = compressor(into{k}, par{k}, itg, d{k});
    end
end