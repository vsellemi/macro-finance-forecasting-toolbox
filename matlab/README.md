

# Overview 

This repository contains sample MATLAB codes for
macroeconomic forecasting. The main files are contained in the `matlab`
directory and are named `main_<model name>.m`. These files require
necessary helper functions contained in the `util` subdirectory and data
files contained in the `data` subdirectory. Users should also ensure
that they have installed the `Econometrics` and
`Statistics and Machine Learning` Toolboxes with MATLAB version
`R2021a+`.

# Data 

For these examples, we use the FRED-MD monthly panel of US macroeconomic
and financial variables from @FRED-MD. Our version of these data span
between 1959:01 - 2021:02 and contain 134 predictors. The folder
`fred_md_preprocessing` contains code from the paper to (i) implement
data transformations to achieve stationarity of all series and (ii)
estimate factors using Principal Components Analysis (PCA) on the panel.

For ease of implementation, the default setting in our forecasting
models is to form predictors as a subset of the first 8 PCA factors and
their lags.[^3] These factors explain over 50% of total variation in
FRED-MD panel. The default is also to include an autoregressive
component in the forecasting target. The folder `data` contains the
FRED-MD stationary-transformed data and the file containing the first 8
PCA factors.

Our main files allow for five different forecasting targets as in
@coulombe2020machine: industrial production (INDPRO), unemployment rate
(UNRATE), consumer price index (CPI), difference between 10-year
treasury rate and federal funds rate (SPREAD), and housing starts
(HOUST). The main files allows users to specify the forecasting target
and forecasting horizon by defining the `YY` and `h` objects,
respectively.

# Main Files

The main files for the forecasting models are named
`main_<model name>.m` and implement selection, estimation, and
evaluation steps for each model.

## Settings 

These files allow users to specify the following settings.

1.  Forecasting target and horizon: `YY` and `h`.

2.  Model selection method: `model_selection`. We split data into three
    equally sized sequential subsets (training, validation, and testing
    samples) and allow users to choose from 4 model selection options.
    The options include:

    -   `ictr`: maximizing in-sample information criteria (BIC, AIC, HQ)
        calculated on the training sample for models estimated on the
        training sample

    -   `icval`: maximizing in-sample information criteria (BIC, AIC,
        HQ) calculated on the validation sample for models estimated on
        the training sample. The specific choice of information criteria
        can be specified via `icidx`.

    -   `pooscv`: pseudo out-of-sample cross-validation with an
        expanding window. This option minimizes pseudo out-of-sample
        forecasting error on the validation sample. This is the most
        computationally intensive option.

    -   `kfcv`: K-fold cross-validation. This option splits the combined
        validation and training samples into K+1 equally size sequential
        subsets (folds) and minimizes the average forecasting error in
        fold $i+1$ from models estimated on folds 1 to $i$ where
        $i=1,...,5$. The number of folds is specified by `K`.

    The default setting here is `icval` with Bayesian Information
    Criteria (BIC).

3.  Out-of-sample procedure: users can specify if they want to do
    out-of-sample forecasting using a rolling or expanding window. The
    default is an expanding window forecast but a rolling window can be
    specified by setting `roll` equal to 1. Window length is set with
    `wL`.

4.  Evaluation benchmark: users can specify the forecasting benchmark
    used in evaluation. The default is a random walk (`RW`) forecast but
    prevailing mean (`PM`) is also supported.

5.  Hyperparameter space: each forecasting model is associated with a
    set of hyperparameters (HPs). Model selection iterates over all
    possible combinations of hyperparameters and users can define this
    set (e.g., maximum number of AR lags, maximum number of factors).

## Evaluation

The main files also provide several standard point forecast evaluation
metrics, namely @DMTest test, @White2000Reality and @Hansen2005test
p-values, and the @Hansen2011MCS model confidence set. These tests are
implemented relative to the specified benchmark. We also provide code to
implement the @MZTest forecast evaluation test. Lastly, users generate
time-series plots of the forecast, cumulative mean squared prediction
error, and rolling root mean squared prediction error of the model and
benchmark.

## Models 

1.  `main_AR.m`: autoregressive model

    -   HPs: lag length (`py`)

2.  `main_ARDI.m`: factor augmented autoregressive model

    -   HPs: AR lag length (`py`), factor lag length (`pf`), number of
        factors (`nf`)

3.  `main_PLS.m`: partial least squares regression

    -   HPs: AR lag length (`py`), factor lag length (`pf`), number of
        factors (`nf`)

4.  `main_ENET.m`: elastic net regression

    -   HPs: AR lag length (`py`), factor lag length (`pf`), number of
        factors (`nf`), lasso penalty (`lambda`), weight of lasso versus
        ridge penalty (`alpha`)

5.  `main_LASSO.m`: lasso regression

    -   HPs: AR lag length (`py`), factor lag length (`pf`), number of
        factors (`nf`), lasso penalty (`lambda`)

6.  `main_KRR.m`: kernel ridge regression

    -   HPs: AR lag length (`py`), factor lag length (`pf`), number of
        factors (`nf`), ridge penalty (`lambda`), kernel function
        (`kernel`), Gaussian (RBF) kernel parameter (`sigma`)

7.  `main_SVR.m`: support vector regression

    -   HPs: AR lag length (`py`), factor lag length (`pf`), number of
        factors (`nf`), kernel function (`kernel`), Gaussian (RBF)
        kernel parameter (`sigma`), SVR optimization parameters
        (`eps_max`, `C_max`)

8.  `main_RF.m`: random forest ensemble regression

    -   HPs: AR lag length (`py`), factor lag length (`pf`), number of
        factors (`nf`), bagging or boosting for aggregation (`method`),
        number of ensemble learning cycles (`numCycles`), resampling
        procedure (`resamp_ind`, `replac_ind` )

9.  `main_NN.m`: feed-forward neural network regression

    -   HPs: AR lag length (`py`), factor lag length (`pf`), number of
        factors (`nf`), hidden unit activation (`hidden_activation`),
        output activation function (`output_activation`), optimizer
        (`solver`), hidden dimension (`hidden_dim`), number of layers
        (`nlayers`)

10. `main_TVPSV.m`: time-varying parameter stochastic volatility model

    -   HPs: see @PettenuzzoTimmermann2015

11. `main_MS.m`: Markov-Switching model

    -   HPs: see @PettenuzzoTimmermann2015

12. `main_CSR.m`: complete subset regression

    -   HPs: see @ELLIOTT2013357
