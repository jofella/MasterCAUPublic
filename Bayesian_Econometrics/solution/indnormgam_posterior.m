function [params, moments] = indnormgam_posterior(y, X, d, beta0, V0, s0_1, nu0_1, s0_2, nu0_2, S0, S1)
% -------------------------------------------------------------------------
% ===| Participant:
% Josef Fella
%
% -------------------------------------------------------------------------
% Exercise 3: [Ref Textbook - p.64 (Gibbs sampling)]
% 
% This function calculates the posterior of the multiple regression model using gibbs sampling
%
% -------------------------------------------------------------------------
% Input: 
% y      (N x 1)  vector of the dependent variable
% X      (N x k)  matrix of exogenous variables
% d      (N x k)  vector of recession dummies
% beta0  (k x 1)  Prior mean for beta (e.g. OLS estimate)
% V0     (k x k)  Prior variance for beta 
% s0_1   (scalar) inverse of the scalar prior mean for h1: s0^2
% nu0_1  (scalar) prior degrees of freedom for h1
% s0_2   (scalar) inverse of the scalar prior mean for h2: s0^2
% nu0_2  (scalar) prior degrees of freedom for h2
% S0     (scalar) burn-in period
% S1     (scalar) posterior sample size
% -------------------------------------------------------------------------
% Output:
% --- "Burnt"-values -- For Convergence Analysis
% params.beta_S0    (S0 x k): burn-in draws of beta
% params.h1_S0      (S0 x 1): burn-in draws of h1
% params.h2_S0      (S0 x 1): burn-in draws of h2
% --- Actual sample:
% params.beta       (S1 x k): posterior draws of beta
% params.h1         (S1 x 1): posterior draws of h1
% params.h2         (S1 x 1): posterior draws of h2
% moments.beta_mean (k x 1):  posterior mean of beta
% moments.beta_std  (k x 1):  posterior standard deviation of beta
% moments.h1_mean   (scalar): posterior mean of h1
% moments.h1_std    (scalar): posterior standard deviation of h1
% moments.h2_mean   (scalar): posterior mean of h2
% moments.h2_std    (scalar): posterior standard deviation of h2
% -------------------------------------------------------------------------
% FOR REFERENCE: "0_..." = prior, "1_..." = posterior
% -------------------------------------------------------------------------

% ===| Set parameters:
k   =    size(X,2);      % Number of regressors
T   =    size(y, 1);     % Sample size/number of days
T2  =    sum(diag(d));   % Number of recession periods, diag(d) is re-transorming Matrix into vector
T1  =    T-T2;           % Number of calm periods


% ===| Gibbs Sampler:
% Recall: A MCMC technique where the joint posterior does not have a
% analytically known form. Hence, we use conditional (and marginal/int.
% out) of the regression coefficient beta and error precisions h1, h2 to
% get conditional posteriors. Those conditionals do have analytical
% suitable forms, so we can use them for sampling. The main idea is to draw
% from the conditinal distributions, but keeping all the other parameters
% (which are conditioned on) fixed. 
% For detailed algor see pseudo code in [Ex.2].


% Intitialize storage matrices for posterior draws:
beta    = zeros(S0+S1, k);
h1      = zeros(S0+S1, 1);
h2      = zeros(S0+S1, 1);

% Initial draw beta(0) using OLS estimates:
beta_initial = mvnrnd(beta0, V0, 1); % Random draw from OLS --> Since conditional is multnormal

% First draw h(1)|beta_intial:
nu1_1 = T1 + nu0_1;
s1_1 = ((y-X*beta_initial')'*diag(1-diag(d))*(y-X*beta_initial') + nu0_1*s0_1) / nu1_1;
h1(1) = gamrnd(nu1_1/2, 2/(nu1_1*s1_1));

% First draw h(2)|beta_intial:
nu1_2 = T2 + nu0_2;
s1_2 = ((y-X*beta_initial')'*d*(y-X*beta_initial') + nu1_2*s0_2) / nu1_2;  % "'" at beta is a technical detail, else dimensions wrong
h2(1) = gamrnd(nu1_2/2, 2/(nu1_2*s1_2));

% First draw beta(1)|h1(1), h1(2):
H         = diag(h1(1) * (1 - diag(d)) + h2(1) * diag(d));
V1        = inv(inv(V0) + X'*H*X);                          % Posterior variance for beta|h1,h2, inverse cause V^-1
beta1     = V1*(inv(V0)*beta0 + X'*H*y);                    % Posterior mean for beta|h1,h2 - "*"V1 cause its a matrix so mult by inverse
beta(1,:) = mvnrnd(beta1,V1,1);                             % Draw from the conditional Normal for beta|h

% Gibbs sampler loop:
for i=2:S0+S1
    % Draw h1(i)|beta(i-1):
    s1_1 = ((y-X*beta(i-1,:)')'*(y-X*beta(i-1,:)') + nu0_1*s0_1)/nu1_1;
    h1(i) = gamrnd(nu1_1/2, 2/(nu1_1*s1_1));
    
    % Draw h2(i)|beta(i-1):
    s2_1 = ((y-X*beta(i-1,:)')'*(y-X*beta(i-1,:)')+nu0_2*s0_2)/nu1_2;
    h2(i) = gamrnd(nu1_2/2, 2/(nu1_2*s2_1));
    
    % Draw beta(i)|h1(i),h2(i)
    H         = diag(h1(i) * (1 - diag(d)) + h2(i) * diag(d));
    V1        = inv(inv(V0) + X'*H*X);
    V1        = (V1 + V1')/2;
    beta1     = V1*(inv(V0)*beta0 + X'*H*y); 
    beta(i,:) = mvnrnd(beta1, V1, 1);   
end


% ===| Store posterior parameters: 
% Recall: We separate the sample in two different part: One is the burn-in
% period (not used for inference) and one our actual sample. Thats because
% MCMC methods, like Gibbs are firstly influence by the intital value. So
% we want to be sure that our markov chain reached stationatity and is not
% influenced by the starting value.

% Save burn-in draws S0 separately and then discard them in beta and h:

% "Burnt"-values - Analyze convergence MC-Chain
params.beta_S0  = beta(1:S0,:); 
params.h1_S0    = h1(1:S0,:);
params.h2_S0    = h2(1:S0,:);

% Actual sample
params.beta     = beta(S0+1:end,:);
params.h1       = h1(S0+1:end,:);
params.h2       = h2(S0+1:end,:);


% ===| Get moments: 

% Posterior mean and variance of beta 
moments.beta_mean = mean(beta);
moments.beta_std = std(beta); 

% Posterior mean and variance of h1
moments.h1_mean = mean(h1);
moments.h1_std = std(h1);

% Posterior mean and variance of h2
moments.h2_mean = mean(h2);
moments.h2_std = std(h2);