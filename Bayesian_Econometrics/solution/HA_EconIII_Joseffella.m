%===========| Home Assignment Econometrics III (WiSe 2024/25) |============
% ===| Participant:
% Josef Fella
% 
% -------------------------------------------------------------------------
% ===| Bayesian estimation of the dynamic linear regression model (1):
% UNEMP_t = mu + alpha_1 * UNEMP_t-1 
%          + beta_1 * INPRO_t-1 + ... + beta_q * INPRO_t-q
%          + gamma_1 * CPI_t-1 + ... + gamma_q * CPI_t-q
%          + phi_1 * BCONF_t-1 + ... + phi_q * BCONF_t-q + lambda * COVID_t + e_t, e_t ~ (0,1/h_i)
% 
% where h_i = h_1 if t is a calm period and h_i = h_2 if t is a recession.
%
% In matrix form:
%
% y = X*beta + e,   e ~ N(0,H^{-1}),
%
% with independent Normal-Gamma prior as follows:
%
% beta ~ N(beta0,1/kappa0)
% h1   ~ G(1/s0_1^2, nu0_1)
% h2   ~ G(1/s0_2^2, nu0_2)
%
% -------------------------------------------------------------------------
% ===| Data description:
% dates      (T-by-1) vector of monthly dates
% data       (T-by-4) matrix of macroeconomic variables for the US economy (stationary series after transformations)
% recessions (J-by-2) matrix of US recessions with start and finish points, where J is the number of dated recessions in the sample 
% -------------------------------------------------------------------------

% ===| Clear data memory:
clear, clc

% ===| Load the data set:
load US_macro.mat

% ===| Plot the transformed (stationary) data:
figure
tiledlayout('flow','TileSpacing','compact','Padding','compact');
for k = 1:size(data,2)
    nexttile, hold on
    plot(dates,data(:,k),'Color','#0072BD','LineWidth',1.5)
    grid on
    yline(0,'LineWidth',1)
    recessionplot
    axis tight
    set(gca,'FontSize',14)
    set(gca,'TickLabelInterpreter','latex','FontSize',18)
    title(VARnames{k},'Interpreter','latex')
end
set(gcf,'Position',[100 100 1400 1000])

% ===| Set baseline specs and adjust the dataset:
q = 4;                                                                      % Set distributed lags
X = [lagmatrix(data(:,1),1) lagmatrix(data(:,2),1:q)...                     % Construct matrix of covariates
      lagmatrix(data(:,3),1:q) lagmatrix(data(:,4),1:q)];
X = X(q+1:end,:);                                                           % Adjust matrix of covariates
y = data(q+1:end,1);                                                        % Set dependent variable and match sample
T = size(y,1);  % sample size N                                                            % Effective sample size
dates = dates(q+1:end);                                                     % Match sample size of dates
COVID = dates >= datetime(2020,04,01) & dates <= datetime(2020,06,01);      % Define COVID-19 dummy variable
X = [ones(T,1) X COVID];                                                    % Matrix of covariates
k = size(X,2);                                                              % Number of covariates
J = size(recessions,1);                                                     % Number of US recessions in the sample
tRecession = zeros(T,1);                                                    % Pre-allocate dummy-vector for recession periods
for j = 1:J
    tRecession = tRecession | (dates >= recessions(j, 1) & dates <= recessions(j, 2)); % Identify dates within the start and end period of each recession
end



%% ===| Excercise 4 - Compute posterior results:

% ===| Set parameters:
k = size(X, 2); % number of regressiors includ. intercept

% ===| Get OLS estimates: (for prior parameters)
XX       = X'*X;    
Xy       = X'*y;    
beta_OLS = X\y;                                         % OLS estimates
ssq_OLS  = ((y-X*beta_OLS)'*(y-X*beta_OLS))/(T-k) ;     % Variance of error term --> =s0_1=s0_2
V_OLS    = ssq_OLS*inv(XX);                             % Variance-covariance matrix of OLS estimates

% ===| Set priors & other input parameters:
beta0   = beta_OLS;
V0      = inv(V_OLS);           % inverse of var/cov matrix
nu0_1   = T-k;              % OLS degree of freedom
s0_1    = ssq_OLS;
nu0_2   = nu0_1;            % Homoscadicity: intial h1=h2
s0_2    = s0_1;
d       = diag(tRecession);     % diagonal Matrix indicating calm/recession period (H-Matrix-Mechanism)
S0      = 1000;
S1      = 10000;

% ===| Compute posterior results via Gibbs sampling:
[p_gibbs, m_gibbs] = indnormgam_posterior(y, X, d, beta0, V0, s0_1, nu0_1, s0_2, nu0_2, S0, S1);


% ===| DISPLAY RESULTS: (somehow the fprint is not working properly for the beta results --> I get some fixed non identifiable results like negative std...)
fprintf(1,'\n\n')
fprintf(1,'--------------------------------------------------------------------------\n')
fprintf(1,'        Posterior results \n\n')

% Display beta results (prior and posterior)
disp('Beta: Mean       Std')
disp('-----------    -------')
disp([m_gibbs.beta_mean', m_gibbs.beta_std']);
fprintf(1,'--------------------------------------------------------------------------\n')
% Display h1 and h2 results (posterior)
disp('Post.Moment     Mean       Std')
disp(' ------------  ------    -------')
fprintf(1,'h1:        %9.2f (%9.2f)\n', m_gibbs.h1_mean, m_gibbs.h1_std);
fprintf(1,'h2:        %9.2f (%9.2f)\n', m_gibbs.h2_mean, m_gibbs.h2_std);

fprintf(1,'---------------------------------------------------------------------------\n')


% ===| Comment on results: | ========================================================
% - After sampling we get the final results in form of mean and variance for our beta and
% h1, h2. Those estimated values can be used to do further infrence or
% forcasting. [see Ex.7]
%
% - Looking at the standard deviation we can see, that with increased sample size it
% slowly decreases. (beta & error precisions) Estimates are getting more precise, which is coherent with theory.
% For example h1 at 10.000 draws has a std of "0.38" and at 1.000.000 draws
% "0.24". 
% Nevertheless, parameter estimates don't vary that much, which
% indicated that a sample size of 10.000 will already give decent results.
% For example beta0 at 10.000 draws we get "-0.0400" and at 1.000.000 "-0.0428". This
% can be observed for all parameters.
% 
% - Burn-in period S0 is set to 1.000 and gibbs-sample size at 10.000.
% Otherwise, computation take quiet long and seemingly doesn't lead to any
% significantly (not in a statistical, but practical sense) better
% estimations. More on burn-in at [Ex.5].
% ======================================================================================


%% ===| Excercise 5 - Check for MCMC-convergence:

% ===| 1.Trace plots:

% beta
names_params = {'$\beta_1$','$\beta_2$','$\beta_3$','$\beta_4$','$\beta_5$','$\beta_6$', ...
    '$\beta_7$','$\beta_8$','$\beta_9$','$\beta_10$','$\beta_11$','$\beta_12$','$\beta_13$'};
figure
sgtitle('Markov Chain Trace Plots','Interpreter','Latex','FontSize',18)
for i = 1:k-2
    subplot(3,5,i)
    plot([p_gibbs.beta_S0(:,i);p_gibbs.beta(:,i)])
    title(names_params{i},'Interpreter','Latex','FontSize',16)
end

subplot(3,5,14)
% error precision h1
plot([p_gibbs.h1_S0(:);p_gibbs.h1(:)])
title('h1','Interpreter','Latex','FontSize',16)

% error precision h2
subplot(3,5,15)
plot([p_gibbs.h2_S0(:);p_gibbs.h2(:)])
title('h2','Interpreter','Latex','FontSize',16)


% ===| Comment on results: | ========================================================
% - Across all parameters, we observe very fast convergence towards a
% (seemingly) stationary process. Additionally, there are no jumps along the Markov
% chains, which gives us confidence that the draws are taken from the
% stationary joint posterior distribution from the start.
%
% - This indicates that using the marginal distribution of beta with OLS estimates
% is a reasonable and effective starting point for our Gibbs sampler.
%
% - The almost instant convergence suggests that S0 (burn-in period) could
% be reduced from 1000 to a smaller value, such as 100 or 200.
%
% - We conclude that our posterior estimates are quite reliable, given
% the fast convergence towards a stationary process and the relatively
% long burn-in period.
%
% - For further research, it would be advisable to carry out a proper
% statistical test, such as CD statistics, to quantitatively assess the
% convergence of the Markov chains.

%======================================================================================


%% ===| Excercise 6 - Compare h1 and h2:

% ===| Comparing h1 and h2: | ========================================================
% - For h1, we observe a mean estimate of 6.22 and a standard deviation of 0.26.
%   For h2, the mean estimate is 3.75 and the standard deviation is 0.20.
%   This indicates that error precision for beta is higher in calm periods (h1)
%   compared to recession or turbulent periods (h2). Economically, this makes
%   sense because employment uncertainty increases and labor markets face
%   greater pressure during recessions.

% - The standard deviation of h2 is lower than that of h1. One explanation is that
%   in "bad" times (recessions), unemployment tends to be consistently high,
%   leading to less variability in the precision estimate (h2). In contrast, during
%   "good" times (calm periods), we may observe both high and low unemployment,
%   reflecting more diversity and resulting in greater uncertainty in the precision
%   estimate (h1). Thus a deeper analysis of this phenomenon should be
%   carried out.

% - Answering the question: Yes, there is empirical evidence that the precision
%   of unemployment rates is lower in recessions. This highlights the importance
%   of differentiating between calm and recession periods; failing to do so
%   could result in less meaningful and reliable model results.
%======================================================================================



%% ===| Excercise 7 - Extend script for forecasting:
% Recall: Here we want to set in a first step our empirical lags as input or basis for our prediction.
% Based on these we will do a one periode forecast for unemployment rate.


% ===| Extract prediction lags: ---> shall be in form like beta
UNEMP_t     = data(end, 1);
INPRO_lags  = flip(data(end-3:end, 2)'); % work around to get row vector and correct order t, t-1, t-2 ...
CPI_lags    = flip(data(end-3:end, 3)');
BCONF_lags  = flip(data(end-3:end, 4)');
COVID_dummy = 0;   % either 0 or 1

% Combine lags into row vector
X_star = [1, UNEMP_t, INPRO_lags, CPI_lags, BCONF_lags, COVID_dummy];


% ===| Perform prediction:

% Assumption no recession (h1): 
y_star_h1 = zeros(S1, 1);
for i = 1:S1
    y_star_h1(i) = normrnd(X_star * p_gibbs.beta(i,:)', 1 / sqrt(p_gibbs.h1(i)));
end

% Assumption recession (h2):
y_star_h2 = zeros(S1, 1);
for i = 1:S1
    y_star_h2(i) = normrnd(X_star * p_gibbs.beta(i,:)', 1 / sqrt(p_gibbs.h2(i)));
end


% ===| Compute statistics:

% h1
y_star_h1_mean = mean(y_star_h1);                 
y_star_h1_sd   = std(y_star_h1);                  
y_star_h1_prc  = prctile(y_star_h1, [2.5, 97.5]); % 95% credible interval

% h2
y_star_h2_mean = mean(y_star_h2);
y_star_h2_sd   = std(y_star_h2);                  
y_star_h2_prc  = prctile(y_star_h2, [2.5, 97.5]); % 95% credible interval


% ===| Display Results:
fprintf(1,'---------------------------------------------------------------------------\n')
disp('Predictive Density Results under h1 and no covid:')
fprintf('Mean: %.4f\n', y_star_h1_mean);
fprintf('Standard Deviation: %.4f\n', y_star_h1_sd);
fprintf('95%% Credible Interval: [%.4f, %.4f]\n', y_star_h1_prc(1), y_star_h1_prc(2));

fprintf(1,'---------------------------------------------------------------------------\n')
disp('Predictive Density Results under h2 and no covid:')
fprintf('Mean: %.4f\n', y_star_h2_mean);
fprintf('Standard Deviation: %.4f\n', y_star_h2_sd);
fprintf('95%% Credible Interval: [%.4f, %.4f]\n', y_star_h2_prc(1), y_star_h2_prc(2));
fprintf(1,'---------------------------------------------------------------------------\n')


% ===| Plot predictive density:

% h1
figure
histfit(y_star_h1, 100, 'normal');
hold on
line([y_star_h1_mean, y_star_h1_mean], ylim, 'LineWidth', 2, 'Color', 'r');        % Predictive mean
line([y_star_h1_prc(1), y_star_h1_prc(1)], ylim, 'LineWidth', 0.5, 'Color', 'g');  % 2.5 percentile
line([y_star_h1_prc(2), y_star_h1_prc(2)], ylim, 'LineWidth', 0.5, 'Color', 'g');  % 97.5 percentile
title('Predictive Density for $UNEMP_{t+1}^*$ under h1', 'Interpreter', 'Latex', 'FontSize', 16);
xlabel('Unemployment Rate (%)', 'Interpreter', 'Latex');
ylabel('Density', 'Interpreter', 'Latex');
legend({'Predictive Density', 'Predictive Mean', '95% Credible Interval'});
hold off

% h2
figure
histfit(y_star_h2, 100, 'normal');
hold on
line([y_star_h2_mean, y_star_h2_mean], ylim, 'LineWidth', 2, 'Color', 'r');        % Predictive mean
line([y_star_h2_prc(1), y_star_h2_prc(1)], ylim, 'LineWidth', 0.5, 'Color', 'g');  % 2.5 percentile
line([y_star_h2_prc(2), y_star_h2_prc(2)], ylim, 'LineWidth', 0.5, 'Color', 'g');  % 97.5 percentile
title('Predictive Density for $UNEMP_{t+1}^*$ under h2', 'Interpreter', 'Latex', 'FontSize', 16);
xlabel('Unemployment Rate (%)', 'Interpreter', 'Latex');
ylabel('Density', 'Interpreter', 'Latex');
legend({'Predictive Density', 'Predictive Mean', '95% Credible Interval'});
hold off