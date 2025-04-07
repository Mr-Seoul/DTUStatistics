import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import sympy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.power as smp
import statsmodels.stats.proportion as smprop

#--------------------------------------------------------------------------------------------------------------------------------#
#Describing Stats

#Describes the data from a SINGLE population
def BasicStatsPopulation(A):
    if (type(A) == list):
        raise ValueError("BasicStatsPopulation: More than one dataset provided, terminating")
    print(f"Count : {len(A)}")
    print(f"Mean : {np.mean(A)}")
    print(f"Std : {np.std(A,ddof=0)}")
    print(f"Variance : {np.var(A,ddof=0)}")
    print(f"Min : {np.percentile(A,0)}")
    print(f"25% : {np.percentile(A,25)}")
    print(f"50% : {np.percentile(A,50)}")
    print(f"75% : {np.percentile(A,75)}")
    print(f"Max : {np.percentile(A,100)}")

#Describes the data from a SINGLE sample
def BasicStatsSample(A):
    if (type(A) == list):
        raise ValueError("BasicStatsSample: More than one dataset provided, terminating")
    print(f"Count : {len(A)}")
    print(f"Mean : {np.mean(A)}")
    print(f"Std : {np.std(A,ddof=1)}")
    print(f"Variance : {np.var(A,ddof=1)}")
    print(f"Min : {np.percentile(A,0)}")
    print(f"25% : {np.percentile(A,25)}")
    print(f"50% : {np.percentile(A,50)}")
    print(f"75% : {np.percentile(A,75)}")
    print(f"Max : {np.percentile(A,100)}")

#Calculates correlation between datasets. The input is a list of np arrays of numbers.
def Correlation(Datasets):
    # Correlation between chosen variables
    df = {}
    for i in range(0,len(Datasets)):
        df.update({i:Datasets[i]})
    data = pd.DataFrame(df)
    print(data.corr())

#--------------------------------------------------------------------------------------------------------------------------------#
#Distributions

#Works for norm, lognorm, exp
#note, SCALE FOR EXP IS 1/LAMBDA 
#Note that Lambda is the average, not the rate parameter. If it lasts 5 months on average, DO NOT PUT expdist(x,0.2), put expdist(x,5)!!!!!!!!!!!!!!!!!!!
#Uniform:
#   args = [loc,scale] loc = mu, scale = height *Note that increasing the scale compresses the function.
#Normal:
#   args = [loc,scale] loc = mu, scale = sigma
#Exponential:
#   args = [loc,scale] loc = 0, scale = 1/Lambda
#Binom:
#    args = [N,p] N=total tries p=probability#
#Poisson:
#    args = [Mu] Mu=1/Successes per month#
#HyperGeometric:
#   args = [M,n,N] M=Total items,n=Total items selected,N=Total succesfull items

#Area of density function
def cdf(distribution,Domain,args):
    dist_input1 = (Domain[1], *args)
    dist_input2 = (Domain[0], *args)

    return distribution.cdf(*dist_input1) - distribution.cdf(*dist_input2)

#Value at points discrete
def pmf(distribution,point,args):
    dist_input = (point, *args)
    return distribution.pmf(*dist_input)

#ppf code for all distributions
def ppf(distribution,point,args):
    dist_input = (point, *args)
    return distribution.ppf(*dist_input)

#Calculate critical values
def Critical_Values(sig,df,tail):
    if tail == "two":
        t_critical = stats.t.ppf(1 - sig/2, df)  # Positive critical value
        return -1*t_critical, t_critical  # Symmetric values for two-tailed test
    elif tail == "right":
        return stats.t.ppf(1 - sig, df)  # One-tailed (right)
    elif tail == "left":
        return stats.t.ppf(sig, df)  # One-tailed (left)
    else:
        raise ValueError("Invalid tail type. Choose 'two', 'left', or 'right'.")

#--------------------------------------------------------------------------------------------------------------------------------#
#Plotting

#Generates a boxplot of a SINGLE dataset
def Boxplot(A, label="", xtext=""):
    if (type(A) == list):
        raise ValueError("Boxplot: More than one dataset provided, terminating")
    
    fig, ax = plt.subplots()

    ax.boxplot(A,label=label)
    plt.xlabel(xtext)
    plt.legend()
    plt.show()

#Generates a histogram of a SINGLE dataset
def Histogram(A,bins=15, label=""):
    if (type(A) == list):
        raise ValueError("Histogram: More than one dataset provided, terminating")
    
    plt.hist(A, bins = bins, color = 'blue', edgecolor = 'black', alpha = 0.5,label=label)
    
    plt.ylabel("Count")
    plt.xlabel("Values")
    plt.legend()
    plt.show()

#Generates a QQplot of a SINGLE dataset
def QQplot(A, label="", xtext=""):
    if (type(A) == list):
        raise ValueError("QQplot: More than one dataset provided, terminating")
    
    sm.qqplot(A, line='q',label=label)
    plt.xlabel(xtext)
    plt.tight_layout()
    plt.legend()
    plt.show()

#Generates the scatter plot of two datasets
def Scatterplot(A,B,label="",xtext="",ytext=""):
    if (type(A) == list or type(B) == list):
        raise ValueError("Scatterplot: More than one dataset provided, terminating")

    plt.scatter(A,B,label=label)
    plt.legend()
    plt.xlabel(xtext)
    plt.ylabel(ytext)
    plt.tight_layout()
    plt.show()

#Generates the wallyplot of a SINGLE dataset
def WallyPlot(A):
    if (type(A) == list):
        raise ValueError("WallyPlot: More than one dataset provided, terminating")

    W = stats.randint.rvs(0,3,size=2)
    # Generate 9 plots with QQ-plots from the assumed distribution
    n = len(A)
    fig, axs = plt.subplots(3,3)
    for ax in axs.flat:
        sm.qqplot(stats.norm.rvs(size=n),line="q",ax=ax)

    axs[W[0],W[1]].clear()
    sm.qqplot((A-A.mean())/A.std(ddof=1),line="q",ax=axs[W[0],W[1]])
    
    temp = plt.setp(axs[W[0],W[1]].spines.values(), color="red")

    plt.tight_layout()
    plt.show()
    return

def LinRegressionPlot(x,y,sig,xtext='x',ytext='y'):
    data = pd.DataFrame({'x': x, 'y': y})
    fit = smf.ols(formula = 'y ~ x', data=data).fit()
    x1_new = pd.DataFrame({'x': np.linspace(min(x), max(x), 100)})
    prediction_summary = fit.get_prediction(x1_new).summary_frame(alpha=sig)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, ' o', label='Observed data')
    plt.plot(x1_new, prediction_summary['mean'], 'r-', label=f"Fitted line (y = {round(fit.params[1],3)}x + {round(fit.params[0],3)}) (R^2 = {round(fit.rsquared,5)})")
    plt.fill_between(x1_new['x'],
    prediction_summary['mean_ci_lower'],
    prediction_summary['mean_ci_upper'],
    color='red', alpha=0.2, label=f'{(1-sig)*100}% Confidence interval')
    plt.fill_between(x1_new['x'],prediction_summary['obs_ci_lower'],prediction_summary['obs_ci_upper'],color='green', alpha=0.2, label=f'{(1-sig)*100}% Prediction interval')
    plt.xlabel(xtext)
    plt.ylabel(ytext)
    plt.legend()
    plt.title(f'Fitted Line with {(1-sig)*100}% Confidence and Prediction Intervals')
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------#
#Confidence intervals

#Generates the confidence interval of the std of a SINGLE dataset
def Norm_CI_Std(A,alpha):
    if (type(A) == list):
        raise ValueError("CI_Std: More than one dataset provided, terminating")
    if (alpha < 0 or alpha > 1):
        raise ValueError("CI_Std: Alpha too large or too small")
    n=A.size
    s = A.std(ddof=1)
    Kai = stats.chi2.ppf([alpha/2,1-alpha/2],df=n-1)

    print(f"{(1-alpha)*100}% CI for std of dataset: ({np.sqrt((n-1)*(s**2)/Kai[1]):.4f}, {np.sqrt((n-1)*(s**2)/Kai[0]):.4f})")

#Generates the confidence interval of the mean of a SINGLE dataset
def Norm_CI_Mean(A,alpha):
    if (type(A) == list):
        raise ValueError("CI_Mean: More than one dataset provided, terminating")
        return
    if (alpha < 0 or alpha > 1):
        raise ValueError("CI_Mean: Alpha too large or too small")
        return
    n=A.size
    mu = A.mean()
    s = A.std(ddof=1)

    print(f"{(1-alpha)*100}% CI for mean of dataset: {stats.t.interval(1-alpha,df=n-1,loc=mu,scale=s/np.sqrt(n))}")

#Generates the confidence interval of the median of a SINGLE dataset
def Norm_CI_Median(A,alpha):
    if (type(A) == list):
        raise ValueError("CI_Median: More than one dataset provided, terminating")
    if (alpha < 0 or alpha > 1):
        raise ValueError("CI_Median: Alpha too large or too small")
    
    A_log = np.log(A)
    n = len(A)
    std_err = np.std(A_log,ddof=1)/np.sqrt(n)

    KI = stats.t.interval(1-alpha, df=n-1, loc=A_log.mean(),scale=std_err)

    print(f"{(1-alpha)*100}% CI for median of dataset: ({np.exp(KI)})")

#--------------------------------------------------------------------------------------------------------------------------------#
#Power

def CalcPower(nobs,sd,delta,sig,ratio,power):
    #Testpower for INDEPENDANT 2 samples
    calcpower = smp.TTestIndPower().solve_power(effect_size=delta/sd,alpha=sig, nobs1=nobs,ratio=ratio)

    return calcpower

def CalcSize(nobs,sd,delta,sig,ratio,power):
    #Testpower for INDEPENDANT 2 samples
    calcsize = smp.TTestIndPower().solve_power(effect_size=delta/sd,alpha=sig, power=power,ratio=ratio)

    return (calcsize,calcsize*ratio)

def CalcMeasureableSize(nobs,sd,delta,sig,ratio,power):
    #Testpower for INDEPENDANT 2 samples
    effect_size = smp.TTestPower().solve_power(nobs=nobs, alpha=sig,power=power,ratio=ratio)

    return effect_size*sd

#--------------------------------------------------------------------------------------------------------------------------------#
#Hypothesis testing

#Calculate t and p for single mean hypothesis
def Hypothesis(A,hyp):
    if (type(A) == list):
        raise ValueError("Hypothesis: More than one dataset provided, terminating")
        return

    res = stats.ttest_1samp(A, popmean = hyp)
    print("t-obs: ", res[0])
    print("p-value: ", res[1])

    # Confidence interval directly from t-test
    print(res.confidence_interval())

#Calculate v,t and p for welchttest
def WelchTest_Equal(A,B):
    if (type(A) == list or type(B) == list):
        raise ValueError("WelchTest: More than one dataset provided, terminating")
    n1 = len(A)
    n2 = len(B)
    s1_sq = np.var(A, ddof=1)
    s2_sq = np.var(B, ddof=1)
    
    df = ((s1_sq / n1 + s2_sq / n2) ** 2) / (
        (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
    )
    res = stats.ttest_ind(A,B, equal_var=False)

    print(f"Degrees of freedom: {df}")
    print(f"t-obs: {res[0]}")
    print(f"p-value: {res[1]}")

#-------------------------------------------------------------------------------------------------------------------------------#
#Non-Parametric Bootstrapping
def NonParametricBootstrap(A,n,alpha):
    X = np.random.choice(A,(n,len(A)),replace=True)
    Xmean = np.mean(X,axis=1)
    Xmedian = np.median(X,axis=1)
    Xstd = np.std(X,axis=1,ddof=1)
    print(f"{1-alpha}% CI mean",np.percentile(Xmean,(alpha/2)*100),np.percentile(Xmean,(1-alpha/2)*100))
    print(f"{1-alpha}% CI median",np.percentile(Xmedian,(alpha/2)*100),np.percentile(Xmedian,(1-alpha/2)*100))
    print(f"{1-alpha}% CI std",np.percentile(Xstd,(alpha/2)*100),np.percentile(Xstd,(1-alpha/2)*100))

#Paremetric Bootstrapping
def ParametricBootstrap(A,n,alpha,distribution):
    args = []
    if (distribution is np.random.normal):
        args = [np.mean(A),np.std(A,ddof=1)]
    elif (distribution is np.random.lognormal):
        args = [np.mean((np.log(A))),np.std(np.log(A),ddof=1)]
    elif (distribution is np.random.exponential):
        args = [np.mean(A)]
    else:
        raise ValueError("Invalid distribution")
    X = distribution(*args,size=(n,len(A)))
    Xmean = np.mean(X,axis=1)
    Xmedian = np.median(X,axis=1)
    Xstd = np.std(X,axis=1,ddof=1)
    print(f"{1-alpha}% CI mean",np.percentile(Xmean,(alpha/2)*100),np.percentile(Xmean,(1-alpha/2)*100))
    print(f"{1-alpha}% CI median",np.percentile(Xmedian,(alpha/2)*100),np.percentile(Xmedian,(1-alpha/2)*100))
    print(f"{1-alpha}% CI std",np.percentile(Xstd,(alpha/2)*100),np.percentile(Xstd,(1-alpha/2)*100))
    
#--------------------------------------------------------------------------------------------------------------------------------#
#Linear regression
def LinRegression(x,y,sig):
    data = pd.DataFrame({'x': x, 'y': y})
    fitData = smf.ols(formula = 'y ~ x', data=data).fit()
    print(fitData.summary(alpha=sig, slim=True))
    print("p-values")
    print(fitData.pvalues)
    print("Error standard deviation")
    # Calculate the residual standard deviation
    error_std = np.sqrt(fitData.mse_resid)
    print(error_std)

def LinPrediction(x,y,sig,val):
    data = pd.DataFrame({'x': x, 'y': y})
    fitData = smf.ols(formula = 'y ~ x', data=data).fit()
    new_data = pd.DataFrame({'x': [val]})
    # Get prediction and confidence intervals
    pd.set_option("display.float_format", None) ## unset option
    pred = fitData.get_prediction(new_data).summary_frame(alpha=sig)
    print(pred)

def MultiLinRegression(XDatasets,y,sig):
    linDict = {'y': y}
    for i in range(0,len(XDatasets)):
        linDict.update({f'x{i}': XDatasets[i]})

    data = pd.DataFrame(linDict)
    Linsentence = "y ~ "
    for i in range(0,len(XDatasets)):
        Linsentence += f"x{i}"
        if i != len(XDatasets) - 1:
            Linsentence += " + "
    
    fitData = smf.ols(formula = Linsentence, data=data).fit()
    print(fitData.summary(alpha=sig, slim=True))
    print("p-values")
    print(fitData.pvalues)
    print("Error standard deviation")
    # Calculate the residual standard deviation
    error_std = np.sqrt(fitData.mse_resid)
    print(error_std)


def MultiLinPrediction(XDatasets,y,sig,vals):
    if (len(XDatasets) != len(vals)):
        raise ValueError("XDatasets or Values have wrong size")
    linDict = {'y': y}
    for i in range(0,len(XDatasets)):
        linDict.update({f'x{i}': XDatasets[i]})

    data = pd.DataFrame(linDict)
    Linsentence = "y ~ "
    for i in range(0,len(XDatasets)):
        Linsentence += f"x{i}"
        if i != len(XDatasets) - 1:
            Linsentence += " + "
    
    fitData = smf.ols(formula = Linsentence, data=data).fit()

    valDict = {f'x{i}': vals[i] for i in range(len(vals))}
    new_data = pd.DataFrame([valDict])  # Single row as a list of dicts

    # Get prediction and confidence intervals
    pd.set_option("display.float_format", None) ## unset option
    pred = fitData.get_prediction(new_data).summary_frame(alpha=sig)
    print(pred)
    
#-----------------------------------------------------------------------------------------------------------------------------------------#'
print("----- Current results start here -----")