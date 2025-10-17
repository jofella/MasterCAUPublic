% =====| Home Assignment Multivariate Time Series Analysis (SoSe 25) |======
%
% - Participant (1): ...
% - Participant (2): Josef Fella
%
% ==========================================================================

% ===| Clear data memory, command window and figures:
clear, close ,clc


%% 1. Inital Data Assessment

% ===| Load the data set (automatically named "data"):
load Germany_Macro_Monthly.mat


% ===| Transform data (seasonal adj.) into stationarity:
xHICP = diff(log(data(:,1)))*100;            % Euro Inflation ex. food/energy (index), WHY log? --> tool to stabilize thing
PROD = diff(log(data(:,2)))*100;             % Industrial production (index)
ORDER = diff(log(data(:,3)))*100;            % Orders received by Industry (index)
BUSEXP = diff(log(data(:,4)))*100;           % IFO business expectations (index)
ENERGY = diff(log(data(:,5)))*100;           % HICP energy price (index)
ECBR = diff(data(:,6));                      % ECB interest rate (ann. percent rate) - no log cause rate itself

data = [xHICP PROD ORDER BUSEXP ENERGY ECBR]; % basically final dataframe with stationary data


% ===| Create quarterly dates from 1960Q2 to 2021Q1:
dates = (datetime(2005,1,2):calmonths(1):datetime(2024,12,1))'; % change "legacy code" to monthly -- review this date section...


% ===| Plot transformed data:
figure
plot(dates,data,'LineWidth',1)
xlabel('Time')
ylabel('m.o.m. change (%)')
title('Transormed EU macro data from 2005M1 to 2024M12')
legend(VARnames{:})
grid on


% =====| Interpretation: | ==================================================
% a) What is the relevance of the stationarity assumption for VAR estimation? [Ref. Lecture 1 - p.39]
%    - Need stable moments in order to do meaningful inference. (nice asympt. properties)
%    - Wold's decomposition theorem: Decomposition of stationary process into 2 uncorrelated processes
%    --> xt = zt + yt with zt deterministic (constant) and yt expressed by MA
%    representation
%    --> If Phi invertable we can express MA as VAR-model: Holds quiet
%    generally for every stationary, nonzero mean nondeterministic process.


% b) What patterns stand out pre- and post-2020?
%   - Pre-Covid:
%       + More or less stationary pattern accros all years.
%       + "Small" dip during financial crisis 2008/09.
%       --> All in all relatively consitent and re-occuring pattern.
%
%   - Post-Covid:
%       + Large downward spike during Covid.
%       + Seems to cause a "break" of known pattern:
%   	--> Somehow still stationary, but swings tend to move
%   	stonger/extremer. (Potentially "new Era"?)


% c) What challenges do energy price volatility and economic turbulence pose for time
%    series models? Which modeling strategies would you suggest to properly account
%    for pandemic effects within the VAR framework?
%   --> structural breaks
%   - One way of handling this is via dummy variables (e.g. 0=pre-covid, 1=post-covid)
%   - Remove covid period.
%   - ...

% =============================================================================


%% 2.1. Model Selection: [Ref. Tut.4/Part1]

% ===| Create dummy variable for Corona (exogenous variable):
COVID_dummy = double(dates >= datetime(2020,03,01) & dates <= datetime(2020,06,01));


% ===| Set VAR parameters:
T    = size(data,1);          % Total sample size
K    = size(data,2);          % Number of variables
pmax = 4;                    % Maximum lag
lags = 1:pmax;                % Lags of the VAR models


% ===| Pre-allocation of VAR objects:
EstMdl(pmax) = varm(K,0);     % Initialize each VAR fit
logL(pmax)   = nan();         % Initialize the logL(iklihood) values

% ===| Fit VAR models and compute information criteria:
for p = 1:pmax
    Mdl = varm(K,p);                                            % Specify VAR(p) structure
    Mdl.SeriesNames = VARnames;                                 % Define variable names
    [EstMdl(p),~,logL(p)] = estimate(Mdl,data,'X',COVID_dummy); % Estimate VAR(p) models and store results + account for ex. dummy
    NumParams = summarize(EstMdl(p)).NumEstimatedParameters;    % Recover number of parameters for each VAR(p)
    [~,~,IC(p)] = aicbic(logL(p), NumParams, T-p);              % Store information criteria
end


% ===| VAR order selection:
IC = squeeze(cell2mat(struct2cell(IC)))';    % Transform IC structure into IC matrix (for better working with)
[ICmin,phat] = min(IC(:,[1 2]));             % Find minimum AIC and BIC (store only nes. values)


% ===| Plot minimum information criteria:
figure
plot(lags',IC(:,[1 2]),'Marker','*','LineWidth',1.5)
hold on
plot(phat',ICmin','Marker','o','MarkerSize',20,'linestyle','none')
xlabel('Lags')
title('Checking VAR Model Adequacy');
legend('AIC','BIC')


%%
% ===| Store BestMdl accroding to AIC-Criterion:
p = 3;                                          % Select optimal p_hat lag order
[BestMdl,~,~,E] = estimate(varm(K,p), data);    % Re-estimate and store selected model
BestMdl.SeriesNames = VARnames;                 % Define variable names


% =====| Interpretation: | ==================================================
% a) Why might AIC and BIC suggest different lag orders?
%    - Its because of there constructive nature, more specifically there
%    penelization terms:
%       * For AIC:
%           - Constant "2K"-term; it prefers fit current data well.
%           - Want to minimize expected prediction error, and penelizes bad
%               approximation/favors greater fit --> Likely "more" lags.
%       * For BIC:
%           - Changes "k*log(T)"-term as sample size increases
%           - Here extra parameters are much harsher penelized/favoring
%               simpler models --> Likely "less" lags.


% b) How would your selection change if your objective were policy analysis versus
%    forecasting? Explain your reasoning based on the bias-variance tradeoff.
%       - Policy/Structual analysis --> Aim: is to precicly capture true
%         dynamic relationships ---> Want: low bias at a cost of potentially
%         higher variance.
%       - Forecasting --> Aim: Minimize out-of-sample prediciton Error (we mostly care about how well we predict 
%         not about "economic meaning") --> Want: Low prediction variance
%         at potentially cost of higher bias.
% have a look at 

% Sidenote: Overfitting with sequential testing. --> Use IC-method.

% =============================================================================


%% ===| 2.2. Check for Granger causality: [Ref. Tut.3]
% Info: Joint Hypo-test on all lagged variables else only first lagged coefficients

gc1 = gctest(BestMdl,'Type','block-wise','Cause',"BUSEXP",'Effect',"PROD"); 
gc2 = gctest(BestMdl,'Type','block-wise','Cause',"PROD",'Effect',"BUSEXP");

% The Granger-cause test says we can excplude PROD in BUSEXP equation,
% but not BUSEXP in PROD equation. Therefore BUSEXP increases forecasting
% accuracy of Production significantly. This also makes sense from a
% theoretical point of view, since only if firms expect good business and
% therefore demand they will produce their goods.



%% 3. Impulse Response Analysis: [Ref. Tut.3]

% ===|set seed:
rng(1)

% ===|Compute IRFS and confidence bounds:
[IRF,Lower,Upper] = irf(BestMdl,'E',E,'NumObs',12,'NumPaths',100,'Confidence',0.95); % Uses cholesky decomp.

% ===| Plot IRFs:
ii=1;
figure
for i= 1:K 
    for j= 1:K
        subplot(K,K,ii)
        plot(IRF(:,j,i),'Color','#0072BD','LineWidth',1.5);
        yline(0);
        hold on
        plot([Lower(:,j,i) Upper(:,j,i)],'b--');
        title(['Effect of shock u_',num2str(j),' on variable y_',num2str(i)]);
        xlabel('Horizon')
        ii=ii+1;
    end
end


% Recall:
% - Columns = what was shocked, -rows = response variable.
% - Effect on next 12 months, use model to ask whats impact of shocks on
%   response varaibles


% =====| Interpretation: | ==================================================
% a) Why the suggested VAR ordering might be a reasonable choice for policy analysis
%    based on a Choleski decomposition?
%     - Idea: Isolated ceteris paribus effect on shocks (usually ut and us are corr = Problem)
%     --> Go from ut to wt, which are uncorrelated, MA representation, VAR ...
%     --> Order matters: Variables only respond contemp. if order before it.
%     --> Here: Prices are on top = In line with academia; Rates are on bottom = 
%         responds to all macro changes contemp. which also makes sense.
%     ---> Seems to be reasonable structure.
%
%
% Ordering matters, we should put the most sluggish moving TS first,
% therefore putting HCI first makes sense, energy prices are basically
% priced continously on exchanges and therefore can be put last


% b) How do shocks to energy prices propagate through the macroeconomic system?
%    Specifically, interpret the transmission of energy shocks to output, using industrial
%    production, orders, and business sentiment. How does monetary policy, proxied
%    by the ECB rate, respond to such shocks?
%     - Here only ECB can react in same period = contemporanious
%     - Other variables will respond with a lag on one period at least
%     - All variables:
%       * Inflation (ex. energy): Up and down , lagged, no significane
%       * Industrial Production: Up, lagged, no significance
%       * Orders: Up, lagged, no significance
%       * Expectations: Up and donw, lagged, no significance
%       * ECB rates: Up, contemporanious, no/small significance

%    - Over time shocks fade away --> coherernt with stationarity/stable
%      model assumption.


% shock effects 
% Inflation: surprisingly only significant one is negative after about 4
% lags
% PRod: significant and fast postive
% Orders : also positive, fast, significant
% EXP: no really big significant effect, a little bit lagged negative
% ECBR : significant positive effect


% Direction: Does it go up or down?
% Timing: Immediate or lagged?
% Persistence: Does the effect fade quickly or last?
% Significance: Do the confidence bands stay away from zero?

% =============================================================================


%% 4. Forecasting Exercise: [Ref. Tut. 4.]

% ===| Set parameters out-of-sample forecasting:
h = 1;                    % One-step ahead forecast horizon
T = 191;                  % Number of consecutive observations for expanding window
N = size(data,1)-T;       % Number of windows // 2021M1 to 2024M12
pmax = 4;                 % Maximum lag order


% ===| Pre-allocation of forecast matrices:
y_hat  = cell(N,pmax);  % Cell structure to store point forecasts
y_MSE  = cell(N,pmax);  % Cell structure to store MSE forecast error matrices
y_FI_l = cell(N,pmax);  % Cell structure to store 95% lower forecast intervals
y_FI_u = cell(N,pmax);  % Cell structure to store 95% upper forecast intervals
FSE    = cell(N,pmax);  % Cell structure to store quadratic forecast errors
% Note that rows refer to the out-of-sample windows while columns to the different VAR(p) models


% ===| Compute out-of-sample forecasts for each VAR(p) model and rolling window:
for p = 1:pmax
    for j = 1:N 
        EstMdl = estimate(varm(K,p),data(1:(T+j-1),:));                        % Estimate the VAR(p) models using expanding window
        [y_hat{j,p}, y_MSE{j,p}] = forecast(EstMdl,h,data((T+j-p):(T+j-1),:)); % Point forecasts and MSE matrix        
        y_FI_l{j,p} = y_hat{j,p} - 2*sqrt(diag(cell2mat(y_MSE{j,p})))';        % Lower forecast bound
        y_FI_u{j,p} = y_hat{j,p} + 2*sqrt(diag(cell2mat(y_MSE{j,p})))';        % Upper forecast bound
        FSE{j,p} = (y_hat{j,p} - data((T+j),:)).^2;                            % Forecast squared errors
    end
end


% ===| Set index for Inflation and Industrial production.
core_idx = 1;
prod_idx = 2;

FSE_mat = cell2mat(FSE);
FSE_mat = reshape(FSE_mat, [N, size(data,2), pmax]);  % N × K × pmax

FSE_selected = FSE_mat(:, 1:2, :);                    % Take only first two variables

MFSE = sqrt(mean(FSE_selected));      % Compute mean forecast squared errors over the N out-of-sample windows
MFSE = mean(reshape(MFSE, [2, pmax])); % Compute final MFSE over the 2 variables in the VAR
[MFSEmin, phat] = min(MFSE); % Find the VAR lag order associated with the minimum MFSE value


% ===| Plot minimum MFSE:
figure
plot(1:pmax,MFSE,'Marker','*','LineWidth',1.5)
hold on
plot(phat,MFSEmin,'Marker','o','MarkerSize',20,'linestyle','none')
xlabel('Lags')
title('Best VAR Forecasting Model');


% ===| Plot out-of-sample forecasts:
y_hat  = cell2mat(y_hat);
y_FI_l = cell2mat(y_FI_l);
y_FI_u = cell2mat(y_FI_u);
y_hat  = reshape(y_hat,[N,K,pmax]);
y_FI_l = reshape(y_FI_l,[N,K,pmax]);
y_FI_u = reshape(y_FI_u,[N,K,pmax]);

figure
t0 = 192;       % Initial date of the plots, start out of sample
for i = 1:K
    subplot(3,2,i)
    h1 = plot(dates(t0:end),data(t0:end,i),'Color','#0072BD','LineWidth',1.2);
    hold on
    h2 = plot(dates(T+1:T+N),y_hat(:,i,phat),'Color','#A2142F','LineWidth',1);
    h3 = plot(dates(T+1:T+N),y_FI_l(:,i,phat),'Color','#A2142F','LineWidth',0.5);
    plot(dates(T+1:T+N),y_FI_u(:,i,phat),'Color','#A2142F','LineWidth',0.2);    
    title(['Out-of-sample VAR(p) forecasts of ' VARnames{i}]);
    h = gca;
    fill([dates(T+1) h.XLim([2 2]) dates(T+1)],h.YLim([1 1 2 2]),'k',...
    'FaceAlpha',0.1,'EdgeColor','none');
    legend([h1 h2 h3],'Observed Values','Point Forecasts','95% Forecast Intervals')
    hold off
end



% =====| Interpretation: | ==================================================
% a) Which VAR model outperforms in terms of root mean squared errors (RMSE) for
%    core inflation and industrial production? Plot the best forecast for core inflation
%    against the actual data over the out-of-sample period.
%    - Best model is VAR(1) based on RMSE.


% b) How do the energy price shocks during this period influence the forecasting performance?
%   - Yes, Forecast accuracy drops during 2022 energy price spike.
%   - Wider forecast intervals, especially for energy & inflation.
%   - Model sometimes misses actual values during shock.
%   - VAR less reliable during high-volatility periods. --> in line with
%     empirics


% =============================================================================