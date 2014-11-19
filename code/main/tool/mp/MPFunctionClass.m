classdef MPFunctionClass < handle
    %MPFUNCTIONCLASS A set of functions to be used with MatchingPursuit.
    %   - The dictionary in matching pursuit is represented with MPDictionary
    %   which contains multiple MPFunctionClass's
    %    
    
    properties(SetAccess=protected)
        % input samples. Instances
        inputInstances;

    end
    
    methods(Abstract)
        % X = input samples. n points.
        % R = dim(output) x n residual matrix.
        %
        % Find and mark the best basis function which reduces total residues 
        % the most. The mark functions will be used to form the final function.
        % Marked function will be excluded from the candidate set (i.e., it will 
        % not be selected again). 
        %
        % Each MPFunctionClass may have its own way of finding the best function 
        % in its class e.g., linear search, gradient descent, etc.
        %
        % return: 
        %   - crossRes = cross correlation between the selected function g and 
        %   the residues in R.
        %   - G a 1xn vector of g(x_i) for all x_i in X
        %   - searchMemento = a state object containing information about the chosen function.
        %   This will be used by markBestBasisFunction(.)
        %   
        [crossRes, G, searchMemento] = findBestBasisFunction(this, R, regParam);

        % Mark the chosen function described in searchMemento. 
        % searchMemento is returned from findBestBasisFunction().
        markBestBasisFunction(this, searchMemento);

        % Return the number of selected (marked) basis functions
        c = countSelectedBases(this);

        % Evaluate the approximated function using the selected basis functions 
        % and stored weights. That is, evaluate sum_i w_i g(x_j) for all j.
        %
        % Return: F (dim(output) x #test points) matrix
        F = evalFunction(this, X);

        % Evaluate marked function on the samples
        % X is an instance of Instances. No W is involved here. Just the selected 
        % basis functions evaluated on X.
        %
        % return: G (#marked x sample size)
        G = evaluate(this, X);
        
        % Evaluate marked function on the inputInstances
        %
        % return: G (#marked x inputInstances count)
        G = evaluateOnTraining(this);
        
        % Evaluate marked function on the subset of inputInstances specified by 
        % the indices.
        %
        % return: G (#marked x length(Ind))
        G = evaluateOnTrainingSubset(this, Ind);

        % Finalize the MPFunctionClass by removing all other candidates the optimize 
        % for evaluation purpose. This method is typically called at the end of 
        % matching pursuit after which new candidates are no longer needed.
        %
        % Return another object of the same class. Do not alter this object.
        obj=finalize(this);

        % The need for this is from backfitting procedure.
        % Set the local weight matrix to the specified W. 
        % Setting an arbitrary W may break the matching pursuit algorithm.
        % Only W resulted from backfitting should be used.
        setWeightMatrix(this, W);
    end
    
end

