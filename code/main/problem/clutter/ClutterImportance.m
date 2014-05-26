classdef ClutterImportance < handle
    %CLUTTERIMPORTANCE Solve clutter problem with importance sampling-based
    %EP.
    
    properties
        w;
        a;
        % proposal distribution for theta
        theta_proposal;
        % importance-sampling size
        iw_samples;
    end
    
    methods
        
        function this=ClutterImportance(a, w, theta_proposal, iw_samples)
            % params. Refer to PRML p.511
            % contamination rate w
            %             this.w = myProcessOptions(op, 'w', 0.5);
            %             this.a = myProcessOptions(op, 'a', 10);
            this.a = a;
            this.w = w;
            assert(isa(theta_proposal, 'Density'));
            assert(isa(theta_proposal, 'Sampler'));
            this.theta_proposal = theta_proposal;
            
            if nargin < 4
                this.iw_samples = 5000;
            else
                this.iw_samples = iw_samples;
            end
            
            
        end
        
        function [R] = ep(this, X, m0, v0, seed)
            a= this.a;
            w=this.w;
            
            oldRs = RandStream.getGlobalStream();
            rs = RandStream.create('mt19937ar','seed',seed);
            RandStream.setGlobalStream(rs);
            
            N = size(X, 2);
            % prior factor for theta
            f0 = DistNormal(m0, v0);
            % product of all incoming messages to theta. Estimate of posterior over
            % theta
            q = f0;
            % f tilde's represented by DistNormal
            FT = DistNormal(zeros(1, N), inf(1, N));
            
            % TM = records of all means of f_i in each iteration
            TM = [];
            % TV = records of variance
            TV = [];
            TMQNI = [];
            TVQNI = [];
            % repeat until convergence
            for t=1:10
                pmean = q.mean;
                pvar = q.variance;
                display(sprintf('## EP iteration %d starts', t));
                for i=1:N
                    
                    qni = q/FT(i); % DistNormal division
                    if ~qni.isproper()
                        display(sprintf('Cavity q\\%d = N(%.2g, %.2g) not proper. Skip.', i, qni.mean, qni.variance));
                        continue;
                    end
                    % we observed X. Use PointMass.
                    %                         mxi_f = PointMass(NX(:,i));
                    % we observed X. But, let's put a width around it
                    mxi_f = DistNormal(X(:,i), 0.1);
                    
                    display(sprintf('x%d = %.2g', i, X(:,i)));
                     
                    qnew=this.project( mxi_f, qni, this.theta_proposal);            
                    if isnan(qnew.mean) 
                        % failed projection
                        display(sprintf('# Projection failed. Skip factor %d', i));
                        continue;
                    elseif  qnew.variance > 0 && qnew.variance < 1e-2
                        qnew = DistNormal(qnew.mean, 1e-2); 
                    end
                    q = qnew;
                    mfi_z = q/qni; %DistNormal division
                    display(sprintf('m_f%d->theta = N(%.2g, %.2g)', i, mfi_z.mean, ...
                        mfi_z.variance));
                    display(sprintf('q = N(%.2g, %.2g)', q.mean, q.variance));
                    FT(i) = mfi_z;
                    TMQNI(t, i) = qni.mean;
                    TVQNI(t, i) = qni.variance;
                    
                    fprintf('\n');
                end
                % object array manipulation
                TM(t, :) = [FT.mean];
                TV(t, :) = [FT.variance];
                % check convergence
                if abs(q.mean-pmean)<1e-2 && abs(q.variance - pvar)<1e-2
                    break;
                end
            end %end main for
            
            R.TM = TM;
            R.TV = TV;
            R.TMQNI = TMQNI;
            R.TVQNI = TVQNI;
            R.q = q;
            R.m = q.mean;
            R.v = q.variance;
            
            RandStream.setGlobalStream(oldRs);
        end
        
        
        function q=project(this, mxi_f, qni, theta_proposal)
            a = this.a;
            w = this.w;
            cond_factor = @(T)(ClutterMinka.x_cond_dist(T, a, w));
            % proposed theta's
            K = this.iw_samples;
            TP = theta_proposal.sampling0(K);            
            XP = cond_factor(TP);
            % compute importance weights
            W = mxi_f.density(XP).*qni.density(TP) ./ theta_proposal.density(TP);
            
            assert( all(W >= 0));
            % projection
            Tsuff = DistNormal.suffStat(TP);
            wsum = sum(W);
            WN = W/wsum;
            ts = Tsuff*WN';
            
            tmean = ts(1);
            tvar = ts(2) - tmean^2;
            
            if all(~isnan(tmean)) && all(~isnan(tvar))
                % W may be numerically 0 if the density values are too low.
                q = DistNormal(tmean, tvar);
          
            else
                q = DistNormal(nan, nan);
            end
            
            
        end
        
        
    end %end methods
    
    methods (Static)
        
    end
    
    
end %end class

