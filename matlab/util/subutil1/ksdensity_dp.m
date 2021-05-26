function ksdensity_dp(Y,varargin)

% Function: Plot kernel density but allows for extra input to control the
% style in the plot

% Error check on Y
if isempty(Y) || size(Y,2) > 1
    error('Y must be a non-empty vector');
end

% Pull x's and y's from Y series
[yy,xx] = ksdensity(Y);

if nargin == 0
    % Simple plot, no frills
    plot(xx,yy);
elseif mod(nargin-1,2) ~= 0
    % Error, extra input arguments needs to be odd number (since extra
    % arguments needs to be pairs of PropertyName (string) and
    % PropertyValue (string or number)
else
    % Pull extra input arguments (PropertyName (string) and
    % PropertyValue (string or number))
    extra_argument = '';
    for i=1:2:nargin-1
        Prop_nam = varargin{i};
        Prop_val = varargin{i+1};
        if ischar(Prop_val)
            extra_argument = [extra_argument,'''',Prop_nam,'''',',','''',Prop_val,'''',','];
        else
            extra_argument = [extra_argument,'''',Prop_nam,'''',',',num2str(Prop_val),','];
        end
    end
    extra_argument(end) = '';
    
    eval(['plot(xx,yy,',extra_argument,');']);
end

        
    
    
    
    