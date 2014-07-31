classdef KProduct < Kernel
    %KPRODUCT A product kernel. 
    % Take multiple other kernels and form a product kernel. This is a meta
    % kernel in the sense that it does nothing by its own. It relies on the
    % specified input kernels.
    
    properties (SetAccess=private)
        % cell array of Kernel's
        kernels;
    end
    
    methods
           
        function this=KProduct(kers)
            % kers is a cell array containing many kernels
            assert(iscell(kers));
            assert(~isempty(kers));
            assert( all(cellfun( @(k)(isa(k, 'Kernel')), kers(:) )) );
            if any(1==size(kers))
                % 1d cell array
                this.kernels = kers(:); % store as a column cell array
            elseif all(size(kers)>=1 ) && length(size(kers))==2
                % 2-dimensional cell array rxc where each row represent a
                % kernel of one type. Construct c product kernels such that
                % each one is formed from r kernels in each column.
                
                for c=1:size(kers, 2)
                    Ks(c) = KProduct(kers(:, c));
                end
                this = Ks;
            else
                error('Invalid input. Must be 1d or 2d cell array of Kernel');
            end
            
            
        end
        
        function Kmat = eval(this, data1, data2)
            % data1 is expected to be a cell array of data sources X={x1,
            % x2, ...x_s} such that each xi can be a input to kernel_i.
%             assert(iscell(data1));
%             assert(iscell(data2));
            assert(length(data1)==length(data2));
            kers = this.kernels;
            assert(length(data1)==length(kers));
            
            % keep in mind Kmat can be huge.
            Kmat = kers{1}.eval(data1{1}, data2{1});
            for i=2:length(kers)
                Ktemp = kers{i}.eval(data1{i}, data2{i});
                % product kernel
                Kmat = Kmat.* Ktemp;
            end
            
        end
        
        function Kvec = pairEval(this, X, Y)
            assert(iscell(X));
            assert(iscell(Y));
            
            evalFunc = @(k, i)(k.pairEval(X{i},Y{i}));
            % cell array Kvecs
            Kvecs= cellfun(evalFunc, this.kernels(:)', ...
                num2cell(1:length(X)), 'UniformOutput', false);
            % each row is a result from one kernel
            R=vertcat(Kvecs{:});
            % return a row vector
            Kvec = prod(R, 1);
           
        end
        
        function Param = getParam(this)
            % return a cell array of cell array (params)
            kers =  this.kernels;
            getpFunc = @(k)(k.getParam());
            Param = cellfun(getpFunc, kers, 'UniformOutput', false);
            
        end
        
        function s=shortSummary(this)
            kers = this.kernels;
            sumFunc = @(k)(k.shortSummary());
            Strs = cellfun(sumFunc, kers, 'UniformOutput', false);
            % add , at the end 
            s = '';
            for i=1:length(Strs)-1
                s = [s, Strs{i}, ', '];
            end

            s = ['{', s, Strs{end}, '}'];
        end
    end
    
    methods (Static)
        
        function Ks=candidates(kers)
            % kers is a 2d cell array of size rxc as in the constructor. 
            % Return a cell array of kernel c candidates.
            Karr = KProduct(kers);
            Ks = num2cell(Karr);
        end
        
        function Ks=cross_product(ker_list1, ker_list2)
            % return a list of kernels Ks of length l1xl2 by considering
            % all combinations of kernels in the two lists.
            % ### Add varargin later for more than two lists #####
            l1 = length(ker_list1);
            l2 = length(ker_list2);
            Ks = cell(l1, l2);
            for i=1:l1
                for j=1:l2
                    k1 = ker_list1{i};
                    k2 = ker_list2{j};
                    Ks{i, j} = KProduct( {k1, k2} );
                end
            end
            Ks = {Ks{:}};
            
        end
    end
    
end

