classdef LDSTransitionBundleGenerator < BundleGenerator & HasOptions
    %LDSTRANSITIONBUNDLEGENERATOR Generate MsgBundle for the factor p(y|x, A)
    %where y is the hidden variabe in the next time step. x is the variable 
    %in the current time step. y ~ N(y; Ax, ep*I). 
    %   - 3 incoming messages from x, y, A. All multivariate Gaussian.
    
    properties (SetAccess=protected)
        % instance of Options 
        options;
    end
    
    methods
        function this=LDSTransitionBundleGenerator()
            this.options = this.getDefaultOptions();
        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';
            kv.iw_samples='Number of importance weights to draw';
            kv.input_dim='Input dimension d for x, y. A is dxd.';

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.iw_samples=3e5;
            st.input_dim=3;

            Op=Options(st);
        end

        % generate a MsgBundle which can be used to train an operator
        function bundle=genBundle(this, n, varOutIndex)
            % index=1 -> y in p(y|x, A)
            assert(ismember(varOutIndex, [1, 2, 3]));
            fbundles = this.genBundles(n);
            bundle = fbundles.getMsgBundle(varOutIndex);

        end

        function fbundles=genBundles(this, n)

            op=this.options.toStruct();
            [ Y, X, A, Yout, Xout, Aout]=LDSTransitionBundleGenerator.genOutBundles(n, op);
            Yda=DistArray(Y);
            Xda=DistArray(X);
            Ada=DistArray(A);
            Youtda=DistArray(Yout);
            Xoutda=DistArray(Xout);
            Aoutda=DistArray(Aout);
            fbundles=FactorBundles({Yda, Xda, Ada}, {Youtda, Xoutda, Aoutda});
        end

        function nv=numVars(this)
            nv=3;
        end

        %s is a string describing this generator.
        function s=shortSummary(this)
            s=mfilename;
        end
    end % end of methods
    
    methods(Static)
        function [ Y, X, A, Yout, Xout, Aout ]=genOutBundles(N, op)
            %Generate training set (messages) for sigmoid factor.
            %
            % T = theta
            % Tout = outgoing messages for theta (after projection)
            % Assume p(x|t) or x=p(t). x is a Beta. t is a Gaussian.
            %
            oldRng = rng();
            rng(op.seed);
            
            d = op.input_dim;
            K = op.iw_samples;
            xproposal = DistNormal(zeros(d, 1), 2*d*eye(d));

            Y = DistNormal.empty(0, 1);
            X = DistNormal.empty(0, 1);
            A = DistNormal.empty(0, 1);
            Yout = DistNormal.empty(0, 1);
            Xout = DistNormal.empty(0, 1);
            Aout = DistNormal.empty(0, 1);
            for i=1:N
                % message from x 
                xmean = randn(d, 1);
                % df (degree of freedom) > d - 1
                xcov= wishrnd(eye(d), d);
                xmsg = DistNormal(xmean, xcov);

                % message from y
                ymean = randn(d, 1);
                ycov = wishrnd(eye(d), d);
                ymsg = DistNormal(ymean, ycov);
                

                % message from A. Flatten form. A should have a narrow 
                % distribution. 
                Amean = randn(d*d, 1);
                Acov = wishrnd(eye(d*d), d*d)/d^2;
                Amsg = DistNormal(Amean, Acov);

                % draw K samples
                XP = xproposal.sampling0(K); % dxK
                AP = Amsg.sampling0(K);
                % multiply AiXi for all i
                B = bsxfun(@times, reshape(AP, [d, d*K]), XP(:)'); %d x dK
                S = sum(reshape(B, [d, d, K]), 2); %d x 1 x K
                AX = squeeze(S); % dxK
                assert(all(size(AX)==[d, K]));

                % sample from factor N(y; Ax, ep*I)
                ep = 0.2;
                YP = mvnrnd(AX', ep*eye(d))'; % d x K
                assert(all(size(YP)==[d, K]));

                % importance weights 
                W = xmsg.density(XP).*ymsg.density(YP)./xproposal.density(XP);
                assert(all(W >= 0));
                wsum = sum(W);
                WN = W/wsum;
                clear W;

                % 2nd moment samples. d^2 x d^2 x K. 
                % Cost much memory.
                AP2 = MatUtils.colOutputProduct(AP, AP);
                Amom2 = reshape(AP2, [d^4, K])*WN';
                clear AP2;
                Amom1 = AP*WN';
                toA = DistNormal(Amom1, reshape(Amom2, [d^2, d^2])-Amom1*Amom1');

                % message to X 
                Xmom1 = XP*WN';
                XP2 = MatUtils.colOutputProduct(XP, XP);
                Xmom2 = reshape(XP2, [d^2, K])*WN';
                clear XP2;
                toX = DistNormal(Xmom1, reshape(Xmom2, [d,d]) - Xmom1*Xmom1');

                % message to Y 
                Ymom1 = YP*WN';
                YP2 = MatUtils.colOutputProduct(YP, YP);
                Ymom2 = reshape(YP2, [d^2, K])*WN';
                clear YP2;
                toY = DistNormal(Ymom1, reshape(Ymom2, [d,d]) - Ymom1*Ymom1');

                if toA.isProper() && toX.isProper() && toY.isProper()
                    % The return message array may not have length N
                    Y(end+1) = ymsg;
                    X(end+1) = xmsg;
                    A(end+1) = Amsg;
                    Yout(end+1) = toY;
                    Xout(end+1) = toX;
                    Aout(end+1) = toA;
                    display(sprintf('%s: generated message pair %d', mfilename, length(X)));
                end
            end
            rng(oldRng);
            assert(length(X)==length(Y));
            assert(length(Y)==length(A));
            assert(length(A)==length(Yout));
            assert(length(Yout)==length(Xout));
            assert(length(Xout)==length(Aout));
        end
    end % end static methods 

end

