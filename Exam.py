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
def Boxplot(A):
    # Validate input: A should be a list/tuple of 1D array-like structures
    if not isinstance(A, (list, tuple)):
        raise ValueError("Boxplot: Input must be a list or tuple of datasets")

    validated = []
    for i, item in enumerate(A):
        arr = np.asarray(item)
        if arr.ndim != 1:
            raise ValueError(f"Boxplot: Dataset at index {i} is not 1D")
        validated.append(arr)

    fig, ax = plt.subplots()

    ax.boxplot(validated,showmeans=True)

    ax.set_title("Boxplots")
    ax.set_xlabel("Dataset Index")
    ax.set_ylabel("Values")

    plt.show()

#Generates a histogram of a SINGLE dataset
def Histogram(A, bins = 15):
    if not isinstance(A, (list, tuple)):
        raise ValueError("Histogram: Input must be a list or tuple of datasets")

    validated = []
    for i, item in enumerate(A):
        arr = np.asarray(item)
        if arr.ndim != 1:
            raise ValueError(f"Histogram: Dataset at index {i} is not 1D")
        validated.append(arr)

    fig, ax = plt.subplots()

    for i, data in enumerate(validated):
        ax.hist(data, bins, alpha=0.5, label=f'Dataset {i+1}')

    ax.set_title("Histograms")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.show()

#Generates a QQplot of a SINGLE dataset
def QQplot(A):
    if (type(A) == list):
        raise ValueError("QQplot: More than one dataset provided, terminating")
    
    sm.qqplot(A, line='q')
    plt.tight_layout()
    plt.show()

#Generates the scatter plot of two datasets
def Scatterplot(A,B):
    if not (isinstance(A, (list, tuple)) and isinstance(B, (list, tuple))):
        raise ValueError("Scatter: X and Y must be lists or tuples of datasets")
    if len(A) != len(B):
        raise ValueError("Scatter: X and Y must contain the same number of datasets")

    validated_x = []
    validated_y = []

    for i in range(len(A)):
        x_arr = np.asarray(A[i])
        y_arr = np.asarray(B[i])
        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError(f"Scatter: Dataset pair at index {i} must be 1D")
        if len(x_arr) != len(y_arr):
            raise ValueError(f"Scatter: Length mismatch in pair at index {i}")
        validated_x.append(x_arr)
        validated_y.append(y_arr)

    fig, ax = plt.subplots()

    for i, (x, y) in enumerate(zip(validated_x, validated_y)):
        ax.scatter(x, y, label=f'Dataset {i+1}', alpha=0.7)

    ax.set_title("Scatter Plots")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

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

def LinRegressionPlot(x,y,sig):
    data = pd.DataFrame({'x': x, 'y': y})
    fit = smf.ols(formula = 'y ~ x', data=data).fit()
    x1_new = pd.DataFrame({'x': np.linspace(0, 1, 100)})
    prediction_summary = fit.get_prediction(x1_new).summary_frame(alpha=sig)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, ' o', label='Observed data')
    plt.plot(x1_new, prediction_summary['mean'], 'r-', label=f"Fitted line (y = {round(fit.params[1],3)}x + {round(fit.params[0],3)}) (R^2 = {round(fit.rsquared,5)})")
    plt.fill_between(x1_new['x'],
    prediction_summary['mean_ci_lower'],
    prediction_summary['mean_ci_upper'],
    color='red', alpha=0.2, label=f'{(1-sig)*100}% Confidence interval')
    plt.fill_between(x1_new['x'],prediction_summary['obs_ci_lower'],prediction_summary['obs_ci_upper'],color='green', alpha=0.2, label=f'{(1-sig)*100}% Prediction interval')
    plt.xlabel('x')
    plt.ylabel('y')
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

    print(f"{(1-alpha)*100}% CI for mean of dataset: {stats.t.interval(
    1-alpha
    ,df=n-1
    ,loc=mu
    ,scale=s/np.sqrt(n))}")

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

def Mean_CI_Single_Sample(A,sig,hyp):
    if (type(A[0]) == list):
        raise ValueError("Hypothesis: More than one dataset provided, terminating")

    res = stats.ttest_1samp(A, popmean = hyp)

    # Confidence interval directly from t-test
    print(res.confidence_interval(1-sig))
    print("If hyp is in range, it is possible")

def Mean_Difference_Two_Samples_CI(A,B,sig):
    if (type(A[0]) == list or type(B[0]) == list):
        raise ValueError("WelchTest: More than one dataset provided, terminating")
    n1 = len(A)
    n2 = len(B)
    s1_sq = np.var(A, ddof=1)
    s2_sq = np.var(B, ddof=1)
    
    df = ((s1_sq / n1 + s2_sq / n2) ** 2) / (
        (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
    )
    res = stats.ttest_ind(A,B, equal_var=False)

    print(res.confidence_interval(1-sig))
    print("If hyp is in range, it is possible")

def Single_Proportion_CI(tot,successful,sig):
    p_hat = successful/tot
    se_p_hat = np.sqrt(p_hat*(1-p_hat)/tot)
    low = tot*p_hat
    high = tot*(1-p_hat)
    if low > 15 and high > 15:
        print("Large Sample")
        print([p_hat - stats.norm.ppf(1-sig/2)*se_p_hat, p_hat + stats.norm.ppf(1-sig/2)*se_p_hat])
    else:
        print("Small Sample, using + 2 method")
        p_hat = (successful+2)/(tot+4)
        se_p_hat = np.sqrt(p_hat*(1-p_hat)/(tot+4))
        print([p_hat - stats.norm.ppf(1-sig/2)*se_p_hat, p_hat + stats.norm.ppf(1-sig/2)*se_p_hat])


#--------------------------------------------------------------------------------------------------------------------------------#
#Power

def CalcPower(nobs,sd,delta,sig,ratio,power):
    #Note that for sig = 0.05, alpha must be 0.95. Thus alpha = 1-sig
    #Testpower for INDEPENDANT 2 samples
    calcpower = smp.TTestIndPower().solve_power(effect_size=delta/sd,alpha=1-sig, nobs1=nobs,ratio=ratio)

    return calcpower

def CalcSize(nobs,sd,delta,sig,ratio,power):
    #Note that for sig = 0.05, alpha must be 0.95. Thus alpha = 1-sig
    #Testpower for INDEPENDANT 2 samples
    calcsize = smp.TTestIndPower().solve_power(effect_size=delta/sd,alpha=1-sig, power=power,ratio=ratio)

    return (calcsize,calcsize*ratio)

def CalcMeasureableSize(nobs,sd,delta,sig,ratio,power):
    #Note that for sig = 0.05, alpha must be 0.95. Thus alpha = 1-sig
    #Testpower for INDEPENDANT 2 samples
    effect_size = smp.TTestPower().solve_power(nobs=nobs, alpha=1-sig,power=power,ratio=ratio)

    return effect_size*sd

#--------------------------------------------------------------------------------------------------------------------------------#
#Hypothesis testing

#Calculate t and p for single mean hypothesis
def Hypothesis(A,hyp,sig):
    if (type(A[0]) == list):
        raise ValueError("Hypothesis: More than one dataset provided, terminating")

    res = stats.ttest_1samp(A, popmean = hyp)
    print("t-obs: ", res[0])
    print("p-value: ", res[1])

#Calculate v,t and p for welchttest
def WelchTest_Equal(A,B):
    if (type(A[0]) == list or type(B[0]) == list):
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

def TableHypo_Test_Equal(A,RowNames,ColumnNames):
    table_data = np.array(A)
    pill_study = pd.DataFrame(table_data, index=RowNames, columns=ColumnNames)
    print(pill_study)
    chi2, p_val, dof, (expected) = stats.chi2_contingency(pill_study, correction=False)
    print
    print("Chai-square test statistic:", chi2)
    print("P-value:", p_val)
    print(f"Dof: {dof}")

def ProportionHypTest(tot,successes,psample,phyp):
    z_obs,p_value = smprop.proportions_ztest(count=successes, nobs=tot, value=psample, prop_var=phyp) 
    # without showing only four decimal places
    print(f"p-value: {2 * stats.norm.cdf(z_obs, loc=0, scale=1)}")


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
    print(round(pred,2))

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
    valDict = {}
    for i in range(0,len(vals)):
        valDict.update({f'x{i}': vals[i]})
    print(valDict)

    new_data = pd.DataFrame(valDict)
    # Get prediction and confidence intervals
    pd.set_option("display.float_format", None) ## unset option
    pred = fitData.get_prediction(new_data).summary_frame(alpha=sig)
    print(round(pred,2))
    
#-----------------------------------------------------------------------------------------------------------------------------------------#'
#Anova testing
def OneWayAnovaTable(A):
    data = {"group":[],"value":[]}
    for i in range(0,len(A)):
        for j in A[i]:
            data["group"].append(str(i))
            data["value"].append(j)
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)
    print(anova_table)

def TwoWayAnovaTable(A):
    data = {"value":[],"group":[],"method":[]}
    for i in range(0,len(A)):
        for j in range(0,len(A[i])):
            data["value"].append(A[i][j])
            data["group"].append(str(i))
            data["method"].append(str(j))
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group) + C(method)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)
    print(anova_table)

def OneWayAnovaCriticalValues(A,sig):
    data = {"group":[],"value":[]}
    for i in range(0,len(A)):
        for j in A[i]:
            data["group"].append(str(i))
            data["value"].append(j)
    df = pd.DataFrame(data)

    dfn = len(A)-1
    dfd = len(data['value'])-len(A)

    fcrit= stats.f.ppf(1-sig, dfn = dfn, dfd = dfd)

    print(f"Degrees of freedom | n : {dfn} , d : {dfd}")
    print(f"F-crit: {fcrit}")

def TwoWayAnovaCriticalValues(A,sig):
    data = {"value":[],"group":[],"method":[]}
    for i in range(0,len(A)):
        for j in range(0,len(A[i])):
            data["value"].append(A[i][j])
            data["group"].append(str(i))
            data["method"].append(str(j))
    df = pd.DataFrame(data)
    print(df)

    dfk = len(A)-1
    dfl = len(A[0]) - 1
    dfd = dfk*dfl

    fcritgroup= stats.f.ppf(1-sig, dfn = dfk, dfd = dfd)
    fcritmethod= stats.f.ppf(1-sig, dfn = dfl, dfd = dfd)

    print(f"Degrees of freedom | dk : {dfk} dl: {dfl}, d : {dfd}")
    print(f"F-crit group: {fcritgroup}")
    print(f"F-crit method: {fcritmethod}")

def SingleOneWayCI_AnovaTest(A,index1,index2,sig,NameArray=[]):
    data = {"group":[],"value":[]}
    for i in range(0,len(A)):
        for j in A[i]:
            data["group"].append(str(i))
            data["value"].append(j)
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)

    if index1 == index2:
        print("Equal indices Error")
        return

    n = len(data['value'])
    k = len(A)

    delta = np.mean(A[index1])-np.mean(A[index2])
    tstat = stats.t.ppf(1-sig/2,n-k)
    SSE = anova_table["sum_sq"]["Residual"]
    ni = len(A[index1])
    nj = len(A[index2])
    lower = delta - tstat*(((SSE/(n-k))*(1/ni + 1/nj))**(1/2))
    upper = delta + tstat*(((SSE/(n-k))*(1/ni + 1/nj))**(1/2))
    if NameArray != []:
        print(f"difference {(1-sig)*100} confidence interval {NameArray[index1]}-{NameArray[index2]}: [ {lower} , {upper} ]")
    else:
        print(f"difference {(1-sig)*100} confidence interval {i}-{j}: [ {lower} , {upper} ]")
    print(f"Do NOT use bernolli's adjusted alpha here: sig = {sig}")

def SingleTwoWayCI_AnovaTest(A,index1,index2,typeTest,sig,GroupNameArray=[],MethodNameArray=[]):
    data = {"value":[],"group":[],"method":[]}
    for i in range(0,len(A)):
        for j in range(0,len(A[i])):
            data["value"].append(A[i][j])
            data["group"].append(str(i))
            data["method"].append(str(j))
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group) + C(method)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)

    if index1 == index2:
        print("Equal indices Error")
        return

    k = len(A)
    l = len(A[0])

    dfl = (k-1)*(l-1)
    if typeTest == "group":
        print("Single CI Group")
        delta = np.mean(A[index1])-np.mean(A[index2])
        tstat = stats.t.ppf(1 - sig/2,dfl)
        SSE = anova_table["sum_sq"]["Residual"]
        ni = len(A[index1])
        nj = len(A[index2])
        lower = delta - tstat*(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
        upper = delta + tstat*(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
        if GroupNameArray != []:
            print(f"difference {(1-sig)*100} confidence interval {GroupNameArray[index1]}-{GroupNameArray[index2]}: [ {lower} , {upper} ]")
        else:
            print(f"difference {(1-sig)*100} confidence interval {index1}-{index2}: [ {lower} , {upper} ]")
    elif typeTest == "method":
        print("CI-Method")
        def meancollumn(arr,index):
            sum = 0
            num = 0
            for subarr in arr:
                if len(subarr) > index:
                    sum += subarr[index]
                    num += 1
            return sum/num
        
        def numcollumn(arr,index):
            sum = 0
            for subarr in arr:
                if len(subarr) > index:
                    sum += 1
            return sum
        
        delta = np.mean(meancollumn(A,index1))-np.mean(meancollumn(A,index2))
        tstat = stats.t.ppf(1-sig/2,dfl)
        SSE = anova_table["sum_sq"]["Residual"]
        ni = numcollumn(A,index1)
        nj = numcollumn(A,index2)
        lower = delta - tstat*(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
        upper = delta + tstat*(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
        if MethodNameArray != []:
            print(f"difference {(1-sig)*100} confidence interval {MethodNameArray[index1]}-{MethodNameArray[index2]}: [ {lower} , {upper} ]")
        else:
            print(f"difference {(1-sig)*100} confidence interval {index1}-{index2}: [ {lower} , {upper} ]")
    else:
        print("Wrong Test Type. Try 'group' or 'method'")
    print(f"Do NOT use bernolli's adjusted alpha here: sig = {sig}")

def AllOneWayCI_AnovaTest(A,sig,NameArray=[]):
    data = {"group":[],"value":[]}
    for i in range(0,len(A)):
        for j in A[i]:
            data["group"].append(str(i))
            data["value"].append(j)
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)

    n = len(data['value'])
    k = len(A)

    for i in range(0,len(A)):
        for j in range(0,len(A)):
            if i != j:
                delta = np.mean(A[i])-np.mean(A[j])
                M = k*(k-1)/2
                bonSig = sig/M
                tstat = stats.t.ppf(1-bonSig/2,n-k)
                SSE = anova_table["sum_sq"]["Residual"]
                ni = len(A[i])
                nj = len(A[j])
                lower = delta - tstat*(((SSE/(n-k))*(1/ni + 1/nj))**(1/2))
                upper = delta + tstat*(((SSE/(n-k))*(1/ni + 1/nj))**(1/2))
                if NameArray != []:
                    print(f"difference {(1-sig)*100} confidence interval {NameArray[i]}-{NameArray[j]}: [ {lower} , {upper} ]")
                else:
                    print(f"difference {(1-sig)*100} confidence interval {i}-{j}: [ {lower} , {upper} ]")
    print(f"BernolliSigma = {bonSig}")

def AllTwoWayCI_AnovaTest(A,sig,GroupNameArray=[],MethodNameArray=[]):
    data = {"value":[],"group":[],"method":[]}
    for i in range(0,len(A)):
        for j in range(0,len(A[i])):
            data["value"].append(A[i][j])
            data["group"].append(str(i))
            data["method"].append(str(j))
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group) + C(method)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)

    k = len(A)
    l = len(A[0])

    dfl = (k-1)*(l-1)
    print("CI - Group")
    for i in range(0,len(A)):
        for j in range(0,len(A)):
            if i != j:
                M = k*(k-1)/2
                bonSig = sig/M
                delta = np.mean(A[i])-np.mean(A[j])
                tstat = stats.t.ppf(1 - bonSig/2,dfl)
                SSE = anova_table["sum_sq"]["Residual"]
                ni = len(A[i])
                nj = len(A[j])
                lower = delta - tstat*(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
                upper = delta + tstat*(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
                if GroupNameArray != []:
                    print(f"difference {(1-sig)*100} confidence interval {GroupNameArray[i]}-{GroupNameArray[j]}: [ {lower} , {upper} ]")
                else:
                    print(f"difference {(1-sig)*100} confidence interval {i}-{j}: [ {lower} , {upper} ]")
    print("CI-Method")
    for i in range(0,len(A[0])):
        for j in range(0,len(A[0])):
            if i != j:
                def meancollumn(arr,index):
                    sum = 0
                    num = 0
                    for subarr in arr:
                        if len(subarr) > index:
                            sum += subarr[index]
                            num += 1
                    return sum/num
                
                def numcollumn(arr,index):
                    sum = 0
                    for subarr in arr:
                        if len(subarr) > index:
                            sum += 1
                    return sum
                
                delta = np.mean(meancollumn(A,i))-np.mean(meancollumn(A,j))
                M = l*(l-1)/2
                bonSig = sig/M
                tstat = stats.t.ppf(1-bonSig/2,dfl)
                SSE = anova_table["sum_sq"]["Residual"]
                ni = numcollumn(A,i)
                nj = numcollumn(A,j)
                lower = delta - tstat*(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
                upper = delta + tstat*(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
                if MethodNameArray != []:
                    print(f"difference {(1-sig)*100} confidence interval {MethodNameArray[i]}-{MethodNameArray[j]}: [ {lower} , {upper} ]")
                else:
                    print(f"difference {(1-sig)*100} confidence interval {i}-{j}: [ {lower} , {upper} ]")
    print(f"BernolliSigma = {bonSig}")

def SingleOneWayAnovaHypTest(A,sig,index1,index2,NameArray=[]):
    data = {"group":[],"value":[]}
    for i in range(0,len(A)):
        for j in A[i]:
            data["group"].append(str(i))
            data["value"].append(j)
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)

    if index1 == index2:
        print("Equal indices Error")
        return

    n = len(data['value'])
    k = len(A)
    
    print("Single Hyp test - Group")
    delta = np.mean(A[index1])-np.mean(A[index2])
    SSE = anova_table["sum_sq"]["Residual"]
    ni = len(A[index1])
    nj = len(A[index2])

    tobs = delta/(((SSE/(n-k))*(1/ni + 1/nj))**(1/2))
    pobs = 2 * (1 - stats.t.cdf(abs(tobs), df=n-k))
    
    if NameArray != []:
        print(f"Hyptest {NameArray[index1]}-{NameArray[index2]}: tobs = {tobs} , pobs = {pobs} ]")
    else:
        print(f"Hyptest {index1}-{index2}: tobs = {tobs} , pobs = {pobs} ]")
    print(f"Do NOT use bernolli's adjusted alpha here: sig = {sig}")

def SingleTwoWayAnovaHypTest(A,sig,index1,index2,testType,GroupNameArray=[],MethodNameArray = []):
    data = {"value":[],"group":[],"method":[]}
    for i in range(0,len(A)):
        for j in range(0,len(A[i])):
            data["value"].append(A[i][j])
            data["group"].append(str(i))
            data["method"].append(str(j))
    df = pd.DataFrame(data)
    print(df)
    fit = smf.ols("value ~ C(group) + C(method)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)

    if index1 == index2:
        print("Equal indices Error")
        return

    k = len(A)
    l = len(A[0])

    dfl = (k-1)*(l-1)
    
    if testType == "group":
        print("Single Hyp test - Group")
        delta = np.mean(A[index1])-np.mean(A[index2])
        SSE = anova_table["sum_sq"]["Residual"]
        ni = len(A[index1])
        nj = len(A[index2])

        tobs = delta/(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
        pobs = 2 * (1 - stats.t.cdf(abs(tobs), df=dfl))
        
        if GroupNameArray != []:
            print(f"Hyptest {GroupNameArray[index1]}-{GroupNameArray[index2]}: tobs = {tobs} , pobs = {pobs} ]")
        else:
            print(f"Hyptest {index1}-{index2}: tobs = {tobs} , pobs = {pobs} ]")
    elif testType == "method":
    
        print("Single Hyp test - Method")
        def meancollumn(arr,index):
            sum = 0
            num = 0
            for subarr in arr:
                if len(subarr) > index:
                    sum += subarr[index]
                    num += 1
            return sum/num
        
        def numcollumn(arr,index):
            sum = 0
            for subarr in arr:
                if len(subarr) > index:
                    sum += 1
            return sum

        delta = meancollumn(A,index1)-meancollumn(A,index2)
        SSE = anova_table["sum_sq"]["Residual"]

        ni = numcollumn(A,i)
        nj = numcollumn(A,j)

        tobs = delta/(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
        pobs = 2 * (1 - stats.t.cdf(abs(tobs), df=dfl))
        
        if MethodNameArray != []:
            print(f"Hyptest {MethodNameArray[i]}-{MethodNameArray[j]}: tobs = {tobs} , pobs = {pobs} ]")
        else:
            print(f"Hyptest {i}-{j}: tobs = {tobs} , pobs = {pobs} ]")
    else:
        print("Wrong Test Type. Try 'group' or 'method'")
    print(f"Do NOT use bernolli's adjusted alpha here: sig = {sig}")

def AllOneWayAnovaHypTest(A,sig,NameArray=[]):
    data = {"group":[],"value":[]}
    for i in range(0,len(A)):
        for j in A[i]:
            data["group"].append(str(i))
            data["value"].append(j)
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)

    n = len(data['value'])
    k = len(A)
    M = (k*(k-1))/2

    for i in range(0,len(A)):
        for j in range(0,len(A)):
            if i != j:
                delta = np.mean(A[i])-np.mean(A[j])
                SSE = anova_table["sum_sq"]["Residual"]
                ni = len(A[i])
                nj = len(A[j])

                tobs = delta/(((SSE/(n-k))*(1/ni + 1/nj))**(1/2))
                pobs = 2 * (1 - stats.t.cdf(abs(tobs), df=n-k))
                
                if NameArray != []:
                    print(f"Hyptest {NameArray[i]}-{NameArray[j]}: tobs = {tobs} , pobs = {pobs} ]")
                else:
                    print(f"Hyptest {i}-{j}: tobs = {tobs} , pobs = {pobs} ]")
    print(f"bernolli alpha: {sig/M}")

def AllTwoWayAnovaHypTest(A,sig,GroupNameArray=[],MethodNameArray = []):
    data = {"value":[],"group":[],"method":[]}
    for i in range(0,len(A)):
        for j in range(0,len(A[i])):
            data["value"].append(A[i][j])
            data["group"].append(str(i))
            data["method"].append(str(j))
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group) + C(method)", data=df).fit()
    anova_table = sm.stats.anova_lm(fit)

    k = len(A)
    l = len(A[0])

    dfl = (k-1)*(l-1)

    print("Groups")
    for i in range(0,len(A)):
        for j in range(0,len(A)):
            if i != j:
                delta = np.mean(A[i])-np.mean(A[j])
                SSE = anova_table["sum_sq"]["Residual"]
                ni = len(A[i])
                nj = len(A[j])

                tobs = delta/(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
                pobs = 2 * (1 - stats.t.cdf(abs(tobs), df=dfl))
                
                if GroupNameArray != []:
                    print(f"Hyptest {GroupNameArray[i]}-{GroupNameArray[j]}: tobs = {tobs} , pobs = {pobs} ]")
                else:
                    print(f"Hyptest {i}-{j}: tobs = {tobs} , pobs = {pobs} ]")
    M = (k*(k-1))/2
    print(f"bernolli alpha: {sig/M}")
    print("")
    print("Methods")
    for i in range(0,len(A[0])):
        for j in range(0,len(A[0])):
            if i != j:
                def meancollumn(arr,index):
                    sum = 0
                    num = 0
                    for subarr in arr:
                        if len(subarr) > index:
                            sum += subarr[index]
                            num += 1
                    return sum/num
                
                def numcollumn(arr,index):
                    sum = 0
                    for subarr in arr:
                        if len(subarr) > index:
                            sum += 1
                    return sum

                delta = meancollumn(A,i)-meancollumn(A,j)
                SSE = anova_table["sum_sq"]["Residual"]

                ni = numcollumn(A,i)
                nj = numcollumn(A,j)

                tobs = delta/(((SSE/(dfl))*(1/ni + 1/nj))**(1/2))
                pobs = 2 * (1 - stats.t.cdf(abs(tobs), df=dfl))
                
                if MethodNameArray != []:
                    print(f"Hyptest {MethodNameArray[i]}-{MethodNameArray[j]}: tobs = {tobs} , pobs = {pobs} ]")
                else:
                    print(f"Hyptest {i}-{j}: tobs = {tobs} , pobs = {pobs} ]")
    M = (l*(l-1))/2
    print(f"bernolli alpha: {sig/M}")

def OneWayAnova_Residual_Analysis(A,sig,NameArray=[]):
    data = {"group":[],"value":[]}
    for i in range(0,len(A)):
        for j in A[i]:
            data["group"].append(str(i))
            data["value"].append(j)
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group)", data=df).fit()

    n = len(data['value'])
    k = len(A)
    Residuals = fit.resid

    print("Normality")
    QQplot(Residuals)

    print("Variance by group")
    for i in range(0,len(A)):
        if NameArray != []:
            print(f" {NameArray[i]}: {np.var(A[i])}")
        else:
            print(f" {i}: {np.var(A[i])}")

def TwoWayAnova_Residual_Analysis(A,sig,GroupNameArray=[],MethodNameArray=[]):
    data = {"value":[],"group":[],"method":[]}
    for i in range(0,len(A)):
        for j in range(0,len(A[i])):
            data["value"].append(A[i][j])
            data["group"].append(str(i))
            data["method"].append(str(j))
    df = pd.DataFrame(data)
    fit = smf.ols("value ~ C(group) + C(method)", data=df).fit()

    n = len(data['value'])
    k = len(A)
    Residuals = fit.resid

    print("Normality")
    QQplot(Residuals)

    print("Variance by group")
    for i in range(0,len(A)):
        if GroupNameArray != []:
            print(f" {GroupNameArray[i]}: {np.var(A[i])}")
        else:
            print(f" {i}: {np.var(A[i])}")
    print("")
    print("Variance by method")
    for i in range(0,len(A[0])):
        def columnArr(OldArray,index):
            arr = []
            for j in range(0,len(OldArray)):
                if len(OldArray[j]) > index:
                    arr.append(OldArray[j][index])
            return arr

        if MethodNameArray != []:
            print(f" {MethodNameArray[i]}: {np.var(columnArr(A,i))}")
        else:
            print(f" {i}: {np.var(columnArr(A,i))}")

#-----------------------------------------------------------------------------------------------------------------------------------------#'
print("----- Current results start here -----")