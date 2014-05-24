classdef CondCholFiniteOut < InstancesMapper
    %CONDCHOLFINITEOUT Generic conditional mean embedding operator using incomplete
    %Cholesky for outputs using finite-dimensional feature maps.
    % C_{Z|X} where Z is the output, X is the input. This class supports
    % multiple inputs by considering them to be from a tensor product
    % space.
    properties (SetAccess=private)
        
        In; %X Instances
        Out; %Z matrix
        kfunc; % Kernel
        R; % Cholesky factor of kernel matrix on In 
        regparam; %regularization parameter
        
        % R*R'. Needed in mapInstances()
%         RRT;
        
        % (Z-Out*R'(RR' + lambda*eye(ra))^-1 R)/lamb. Needed in mapInstances()
        ZOutR3;
        
    
    end
    
    methods
        
        function this = CondCholFiniteOut(R, In, Out, kfunc, lambda)
            % R = Cholesky factorization of the kernel matrix on the
            % inputs. Assume R is rxn.
            % In = Instances used to compute K with kfunc whose Cholesky
            % factorization is R'*R.
            % Out = dz x n data matrix of mapped (with finite-dimensional feature map)
            % output variable.
            % kfunc = a Kernel
            assert(isa(In, 'Instances'));
            assert(isnumeric(Out));
            assert(isnumeric(R));
            assert(size(R, 2)==size(Out, 2));
            assert(size(Out, 2)==In.count());
            assert(isa(kfunc, 'Kernel'));
            assert(isnumeric(lambda) && lambda >= 0);
            
            this.In = In;
            this.Out = Out;
            this.kfunc = kfunc;
            this.R = R;
            this.regparam = lambda;
            
            % L, U factor for (this.RRT + lamb*eye(ra)).
            ra = size(R, 1);
            this.ZOutR3 = (Out - ( (Out*R')/(R*R'+lambda*eye(ra)) )*R)/lambda;
        end
        
        
        function Zout = mapInstances(this, Xin)
            % Map Instances in Xin to Zout with this operator.
            assert(isa(Xin, 'Instances'));
            R = this.R;
            In = this.In;
            Z = this.Out;
            kfunc = this.kfunc;
            Krs = kfunc.eval(In.getAll(), Xin.getAll());
            
%             B = (R*R' + lamb*eye(ra)) \ RKrs;
%             B = (this.RRT + lamb*eye(ra)) \ RKrs;
%             y = linsolve(this.L, this.P*RKrs, struct('LT', true));
%             B = linsolve(this.U, y, struct('UT', true));
%             Zout = (Z*Krs - (Z*R')*B)/lamb;
            Zout = this.ZOutR3*Krs;
        end
        
    end %end methods
    
    methods (Static)
        
        function [Op, C] = learn_operator(In, Out,  op)
            assert(isa(In, 'Instances'));
            [ C] = cond_ho_finiteout( In, Out, op );
            ichol = C.bkernel_ichol;
            kfunc = C.bkernel;
            lambda = C.blambda;
            Op = CondCholFiniteOut(ichol.R, In, Out, kfunc, lambda);
        end
        
      
    end
    
end

