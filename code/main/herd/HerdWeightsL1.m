classdef HerdWeightsL1 < HasOptions & InstancesMapper
    %HERDWEIGHTS Learn weights for sending project EP messages by herding
    %   - L1 penalized least squares
    
    properties(SetAccess = protected)
        % An instance of Options
        options;
    end
    
    methods
        function this = HerdWeightsL1()
            this.options = this.getDefaultOptions();
        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;

            Op=Options(st);
        end

        function Zout = mapInstances(this, Xin)

        end

        % return a short summary of this mapper
        function s = shortSummary(this)
            s = sprintf('%s', mfilename);
        end
    end % end methods
    
    methods (Static)

    end
end

