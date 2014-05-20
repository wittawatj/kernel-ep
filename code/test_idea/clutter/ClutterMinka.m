classdef ClutterMinka < handle
    % Clutter problem solved with original EP
    %
    
    properties
        w;
        a;
    end
    
    methods
        
        function this=ClutterMinka(a, w)
            % params. Refer to PRML p.511
            % contamination rate w
            %             this.w = myProcessOptions(op, 'w', 0.5);
            %             this.a = myProcessOptions(op, 'a', 10);
            this.a = a;
            this.w = w;
            
        end
        
        function [R] = ep(this, X, m0, v0, seed)
            a= this.a;
            w=this.w;
            
            oldRs = RandStream.getGlobalStream();
            rs = RandStream.create('mt19937ar','seed',seed);
            RandStream.setGlobalStream(rs);
            
            N = size(X, 2);
            M = zeros(1, N);
            V = inf(1, N);
            S = zeros(1, N);
            MQNI = zeros(1, N);
            VQNI = zeros(1, N);
            % parameter of q
            m = m0;
            v = v0;
            % original EP
            
            % records of change of M and V in every iteration
            TM = [];
            TV = [];
            TS = [];
            % mean of cavity
            TMQNI = [];
            TVQNI = [];
            
            for t=1:100 %EP iterations
                mprev = m;
                vprev = v;
                for i=1:N
                    x_i = X(:, i);
                    v_i = V(:, i);
                    % perturb for stability. No. significantly change the
                    % solution.
                    v_not_i = (1/v - 1/v_i)^-1;
                    %                     v_not_i = (1/v - 1/v_i + 1e-3)^-1 + 1e-3;
                    m_i = M(:, i);
                    m_not_i = m + v_not_i*v_i^-1*(m-m_i);
                    
                    z_i = (1-w)*normpdf(x_i, m_not_i, sqrt(v_not_i+1) ) ...
                        + w*normpdf(x_i, 0, sqrt(a));
                    
                    rho_i = 1 - w*normpdf(x_i, 0, sqrt(a))/z_i;
                    m_new = m_not_i + rho_i*v_not_i*(x_i - m_not_i)/(v_not_i+1);
                    v_new = v_not_i - rho_i*v_not_i^2/(v_not_i+1) ...
                        + rho_i*(1-rho_i)*v_not_i^2*(x_i - m_not_i)^2/((v_not_i + 1)^2);
                    
                    V(:, i) = (v_new^-1 - v_not_i^-1)^-1;
                    if isinf(V(:, i) )
                        V(:,i) = 1e4;
                    end
                    % perturb for stability
                    %                       V(:, i) = (v_new^-1 - v_not_i^-1 + 1e-3)^-1 + 1e-3;
                    M(:, i) = m_not_i + (V(:, i) + v_not_i)*(m_new-m_not_i)/v_not_i;
                    
                    if isnan(M(:,i))
                        display(sprintf('f_tilde_%d: nan mean', i));
                    end
                    
                    if isnan(m_new) || isnan(v_new)
                        continue
                    end
                    s_i = z_i/(sqrt(2*pi*V(:,i))*normpdf(M(:,i), sqrt(V(:,i)+v_not_i) ));
                    S(:, i) = s_i;
                    
                    m = m_new;
                    v = v_new;
                    
                    MQNI(:, i) = m_not_i;
                    VQNI(:, i) = v_not_i;
                    
                end
                TM(t, :) = M;
                TV(t, :) = V;
                TS(t, :) = S;
                TMQNI(t ,:) = MQNI;
                TVQNI(t, :) = VQNI;
                %                 if any(isnan(M)) || any(isnan(V)) || ...
                
                if                        (abs(m-mprev) < 1e-2 && abs(v-vprev) < 1e-2)
                    % need to stop the iterations if there is NaN in any
                    % parameters of f_i. NaN will be propagated in the
                    % following iterations, blowing up.
                    break;
                end
                
                
            end % ep iterations
            R.m = m; %mean of q
            R.v = v; %var of q
            R.TM = TM;
            R.TV = TV;
            R.TS = TS;
            R.TMQNI = TMQNI;
            R.TVQNI = TVQNI;
            R.X = X;
            
            RandStream.setGlobalStream(oldRs);
            
        end
        
    end %end methods
    
    methods (Static)
        
        
        function [X, ftrue, feval] = x_cond_dist(Theta, a, w)
            % Draw X|theta.
            % feval takes to argument (x, theta) and evaluates the factor
            N  = length(Theta);
            cov(:,:,1) = 1;
            cov(:,:,2) = a;
            
            X = zeros(1, N);
   
            % poor performance
%             for i=1:N
%                 theta = Theta(:,i);
%                 f = gmdistribution([theta; 0], cov, [1-w, w]);
%                 X(:,i) = f.random(1);
%             end

            % my own. 
            % Sample mixture components
            Z = 1+ (rand(1, N) > 1-w);
            % theta component
            I1 = Z==1;
            I2 = Z==2;
            if sum(I1)>0
                X(I1) = randn(1, sum(I1) ) + Theta(I1);
            end
            % noise component
            X(I2) = randn(1, sum(I2))*sqrt(a);
            
            ftrue =  gmdistribution([mean(Theta); 0], cov, [1-w, w]);
            feval = @(x, the)( ClutterMinka.factor_eval(x, the, [1,a], w) );
        end
        
        function E=factor_eval(x, theta, vars, w)
            % assume 1d Gaussians
            assert(size(x,2)==size(theta,2));
            
            C1 = (1/(sqrt(2*pi*vars(1))))*exp(-(x-theta ).^2/(2*vars(1)) );
            C2 = normpdf(x, 0, sqrt(vars(2)) );
            E = (1-w)*C1 + w*C2;
        end
        
    end
    
end %end class

