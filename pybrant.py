
"""BrantPy: Implementation of Brant's Test for Parallel Regression Assumption in Ordinal Models

This class implements Brant's (1990) test for the parallel regression assumption 
(proportional odds assumption) in ordinal logistic regression. The test examines whether 
the relationship between pairs of outcome categories is consistent across the range of 
the predictor variables - a fundamental assumption of ordinal regression models.

The implementation extends statsmodels' GenericLikelihoodModel and is compatible with 
statsmodels' OrderedModel fitted using formula or array interfaces.

Mathematical Details:
-------------------
The test works by:
1. Fitting separate binary logistic models at each cut-point of the ordinal outcome
2. Computing variance-covariance matrices for the estimated parameters
3. Constructing chi-square test statistics for the equality of coefficients
4. Testing both globally (omnibus) and individually for each variable

Parameters
----------
model : OrderedResults
    A fitted ordinal regression model from statsmodels
with_formula : bool, default=True
    Whether the model was fitted using statsmodels' formula interface
by_var : bool, optional
    If True, combines coefficient tests for categorical variables
    If False, tests each coefficient separately
**kwds : dict
    Additional keywords passed to GenericLikelihoodModel

Returns
-------
DataFrame
    Contains test results with columns:
    - Test for: Variable name (or 'Omnibus' for global test)
    - X2: Chi-square test statistic
    - df: Degrees of freedom
    - probability: P-value

Methods
-------
TestResults()
    Performs Brant test and returns results DataFrame
_model()
    Internal method to extract model dimensions and data
_temp_data()
    Creates binary outcomes for each cut-point
_store_model()
    Fits separate binary logistic models
_pi()
    Computes predicted probabilities
_X()
    Prepares design matrix
_varBeta()
    Computes variance-covariance matrix
_betaStar()
    Combines coefficient estimates
_X2()
    Calculates chi-square statistics
_combined_results()
    Processes results for categorical variables
_byVar()
    Computes test statistics by variable

Notes
-----
- Requires statsmodels' OrderedModel fitted objects
- Small p-values suggest violation of parallel regression assumption
- The omnibus test examines the assumption globally
- Variable-specific tests help identify problematic predictors
- For categorical variables, by_var=True provides a single test combining all levels

Examples
--------
>>> from statsmodels.miscmodels.ordinal_model import OrderedModel
>>> # Fit ordinal model
>>> ord_model = OrderedModel.from_formula("y ~ x1 + x2", data=df)
>>> fit = ord_model.fit()
>>> # Perform Brant test
>>> brant_test = BrantPy(fit, by_var=True)
>>> results = brant_test.TestResults()
>>> print(results)  # View test results

References
----------
Brant, R. (1990). Assessing proportionality in the proportional odds model for 
ordinal logistic regression. Biometrics, 46(4), 1171-1178.

Implementation inspired by R's brant test:
https://benjaminschlegel.ch/r/brant/

See Also
--------
statsmodels.miscmodels.ordinal_model.OrderedModel
statsmodels.base.model.GenericLikelihoodModel
"""

import numpy as np
import pandas as pd
from scipy.linalg import inv
from pandas.core.common import flatten
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import (OrderedModel,
                                                  OrderedResults)
import scipy.stats as stats
from scipy import linalg
from scipy.linalg import block_diag
from scipy.stats import chi2
import warnings
from statsmodels.base.model import GenericLikelihoodModel
    # Model,
#     LikelihoodModel,
    
#     GenericLikelihoodModelResults
# )

import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm


class BrantPy(GenericLikelihoodModel):

    def __init__(self, model = None , with_formula = True, by_var = None, **kwds):
        
        self.model = model
        self.by_var = by_var
        self.with_formula = with_formula
        if self.with_formula:
            self.var_names = self.model.model.design_info.column_names 
        else:
            self.var_names = list(self.model.model.data.orig_exog.columns)
        
    def _model(self):
        
        model = self.model
    
        df_x = pd.DataFrame(data = model.exog, columns =self.var_names)  
        K = df_x.shape[1]

        df_y = pd.DataFrame({'y' : model.endog})
        J = len(np.unique(df_y['y'].values).tolist())
        
        return J, K, df_x, df_y

    def _temp_data(self):
        model = self.model
        
        J,K,df_x, df_y = self._model()

        df_z = pd.DataFrame(data = np.zeros((len(df_y),J)),
                                columns = [f'z_{m}' for m in range(J)])
        for m in range(J):
            df_z.loc[df_y['y'] > m,f'z_{m}']=1

        temp_data = pd.concat([df_z,df_x],axis =1)

        return J,K,temp_data

    def _store_model(self):
        model = self.model
        
        J, K,temp_data = self._temp_data()

        binary_models = {}
        beta_hat = {}
        var_hat = {}
        for m in range(J-1):
            mod = sm.Logit(endog = temp_data[f'z_{m}'],
                            exog = model.exog).fit(method='bfgs',disp=False,
                                                  maxiter=1000)
            binary_models[f'model{m}'] = mod
            beta_hat[f'model{m}'] = mod.params.values
            var_hat[f'model{m}'] = mod.cov_params().values

        return J, K, binary_models,var_hat,beta_hat

    def _pi(self):
        model = self.model
        
        J, K, binary_models,var_hat,beta_hat = self._store_model()

        pi_hat = {}
        for m in range(J-1):
            pi_hat[f'model{m}'] = binary_models[f'model{m}'].predict()

        return J,K, pi_hat

    def _X(self):
        model = self.model
        
        J, K, df_x, df_y = self._model()
        
        df_X = pd.concat([pd.DataFrame({'Intercept': np.ones(model.nobs)}),
                         df_x], axis=1)
        return df_X
    
    def _varBeta(self):
        model = self.model
        
        J, K, pi_hat = self._pi()
        X = self._X().values
        
        varBeta = np.empty(((J-1)*K,(J-1)*K))
        
        for m in range(J-2):
            for l in range(m+1,J-1):
                Wml = np.diag(pi_hat[f'model{l}'] - pi_hat[f'model{m}']*pi_hat[f'model{l}'])
                Wm = np.diag(pi_hat[f'model{m}'] - pi_hat[f'model{m}']*pi_hat[f'model{m}'])
                Wl = np.diag(pi_hat[f'model{l}'] - pi_hat[f'model{l}']*pi_hat[f'model{l}'])
                MAT = (inv(X.T @ Wm @X) @ (X.T@ Wml @X) @ inv(X.T @ Wl @ X))[1:,1:]
                varBeta[m*K:(m+1)*K,l*K:(l+1)*K] = MAT
                varBeta[l*K:(l+1)*K,m*K:(m+1)*K] = MAT

        return varBeta

    def _betaStar(self):
        model = self.model
        
        J, K, binary_models,var_hat,beta_hat = self._store_model()
        varBeta = self._varBeta()
        
        beta = []
        for m in range(J-1):
            beta.append(beta_hat[f'model{m}']) #[1:])
            varBeta[m*K+1:(m+1)*K,m*K+1:(m+1)*K] = var_hat[f'model{m}'][1:,1:]
       
        betaStar = np.hstack(beta)
        
        return J, K, betaStar, varBeta
    
    def _X2(self):
        model = self.model
        
        J, K, betaStar, varBeta = self._betaStar()

        I = np.eye(K)
        blocks = [-I for _ in range(J-2)]
        offset = 0
        aux = np.empty((0, offset), int)
        A = block_diag(aux, *blocks, aux.T)
        B = np.vstack(np.tile(I,J-2).T) 
        D = np.hstack((B,A))
        DD = np.dot(D,betaStar)
        DV = np.dot(np.dot(D,varBeta),D.T)
        
        X2 = np.asarray([np.dot(np.dot(DD.T,inv(DV)),DD)])
        
        return X2, D

    def _combined_results(self):
        model = self.model
        if self.with_formula: 
            factors = []
            for i,name in enumerate(list(model.model.design_info.term_names)):
                vtype = list(model.model.design_info.factor_infos.values())[i].type
                if vtype == 'categorical':
                    factors.append(name)

                var_names = list(model.model.design_info.term_names)
        else:
            var_names = self.var_names
        

        var = []
        nvar = []   
        for v,variable in enumerate(var_names):
            if self.with_formula: 
                if np.isin(variable,factors):
                    n = len(list(model.model.design_info.factor_infos.values())[v].categories)
                    var.append(np.tile(n-1,n-1).tolist()) 
                    nvar.append(np.tile(variable,n-1).tolist())
                else:
                    var.append([1])
                    nvar.append([variable])
            else:
                var.append([1])
                nvar.append([variable])

            vl =[e for innerList in var for e in innerList]
            nl =[e for innerList in nvar for e in innerList]

        df = pd.concat([
            pd.DataFrame({'icol':np.arange(len(vl))}),
            pd.DataFrame({'term': nl}),
            pd.DataFrame({'k_level': vl})],
                        axis=1).reset_index(drop=True)

        return  df
    
    def _byVar(self):
        
        model = self.model
        by_var = self.by_var
        
        X2, D = self._X2()
        J, K, betaStar, varBeta = self._betaStar()
        
        
        if by_var:
            result = self._combined_results()
            df_vl = []
            for ku in result['term'].unique().tolist():

                kk = result.loc[result['term'] == ku,:]['icol'].values.tolist()   
                sl = []
                df_vtemp = 0
                for k in kk:
                    sl.append(np.arange(k,K*(J-1),K).tolist())
                    df_vtemp += J-2

                df_vl.append(df_vtemp)

                s=[e for innerList in sl for e in innerList]
                s = sorted(s)
                Ds = D[:,s]

                if np.shape(Ds)!=0:
                    Ds = Ds[~np.all(Ds == 0, axis=1)]

                if np.shape(Ds)!=0:
                    X2 = np.vstack((X2,(Ds@betaStar[s]).T @ 
                        inv(Ds @ varBeta[s,:][:,s]@ Ds.T) @ (Ds @ betaStar[s])))
                else:
                    X2 = np.vstack((X2,(Ds@betaStar[s]).T @  
                        inv(Ds @ varBeta[s,:][:,s]@ (Ds.T).T) @ (Ds @ betaStar[s])))
            
            res =  pd.DataFrame({'df': df_vl})
                                
        else:
            
            df_vl = []
            for k in range(K):
                s = np.arange(k,K*(J-1),K)
               
                Ds = D[:,s]

                if np.shape(Ds)!=0:
                    Ds = Ds[~np.all(Ds == 0, axis=1)]

                if np.shape(Ds)!=0:
                    X2 = np.vstack((X2,(Ds@betaStar[s]).T @ 
                            inv((Ds @ varBeta[s,:][:,s])@ Ds.T) @ (Ds @ betaStar[s])))
                else:
                    X2 = np.vstack((X2,(Ds@betaStar[s]).T @  
                        inv((Ds @ varBeta[s,:][:,s]) @ (Ds.T).T) @  (Ds @ betaStar[s])))

                df_vl.append(J-2)
                
            res = pd.DataFrame({'df': df_vl})

        return J, K, res, X2


    # def TestResults0(self):
    #     """Generate formatted results of the Brant test for parallel regression assumption.
        
    #     Returns
    #     -------
    #     str
    #         A formatted string containing test results and interpretation
            
    #     Examples
    #     --------
    #     >>> brant = BrantPy(model, by_var=True)
    #     >>> print(brant.TestResults())
    #     """
    #     print(" model stands for fitted Ordered model .....")

    #     # Get model components
    #     model = self.model
    #     by_var = self.by_var
        
    #     # Calculate test statistics
    #     J, K, df, X2 = self._byVar()
        
    #     # Calculate degrees of freedom
    #     df_v = pd.DataFrame({'df': [(J-2)*K]})
    #     dff = pd.concat([df_v, df], axis=0).reset_index(drop=True)
        
    #     # Calculate p-values
    #     p_value = chi2.sf(X2, dff.values)
        
    #     # Get variable names
    #     if self.with_formula:
    #         if by_var:
    #             var_names = model.model.design_info.term_names
    #         else:
    #             var_names = model.model.design_info.column_names
    #     else:
    #         var_names = self.var_names
        
    #     v_names = ['Omnibus'] + var_names
        
    #     # Create results DataFrame
    #     results = pd.concat([
    #         pd.DataFrame({'Test for': v_names}),
    #         pd.DataFrame({'X2': X2.reshape((-1,))}).round(3),
    #         dff,
    #         pd.DataFrame({'probability': p_value.reshape((-1,))}).round(3)
    #     ], axis=1)
        
    #     # Format header section
    #     model_type = "Formula-based" if self.with_formula else "Array-based"
    #     header = [
    #         "\nBrant Test of Parallel Regression Assumption",
    #         "=" * 50,
    #         f"Model type: {model_type}",
    #         f"Number of observations: {model.nobs}",
    #         f"Number of predictors: {len(var_names)}",
    #         f"Number of outcome levels: {len(np.unique(model.endog))}",
    #         "-" * 50,
    #         "\nTest Results:",
    #         ""
    #     ]
        
    #     # Rename columns for better readability
    #     results = results.rename(columns={
    #         'Test for': 'Variable',
    #         'X2': 'Chi-square',
    #         'df': 'DF',
    #         'probability': 'P-value'
    #     })
    
    #     # Convert results to formatted string
    #     results_str = results.to_string(
    #         index=False,
    #         float_format=lambda x: f"{x:.3f}",
    #         justify='left'
    #     )
        
    #     # Add interpretation section
    #     interpretation = [
    #         "",
    #         "-" * 50,
    #         "\nInterpretation:",
    #         "* H0: Parallel Regression Assumption holds",
    #         "* p-value < 0.05 suggests violation of the parallel regression assumption",
    #         "* Omnibus test examines global validity of the assumption",
    #         "* Individual tests identify specific problematic variables",
    #         ""
    #     ]
        
    #     # Combine all sections
    #     formatted_output = "\n".join(header + [results_str] + interpretation)
        
    #     return formatted_output

    def TestResults(self):
        """Generate results of the Brant test as a DataFrame.
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing test results
        """
        
        # Get model components
        model = self.model
        by_var = self.by_var
        
        # Calculate test statistics
        J, K, df, X2 = self._byVar()
        
        # Calculate degrees of freedom
        df_v = pd.DataFrame({'df': [(J-2)*K]})
        dff = pd.concat([df_v, df], axis=0).reset_index(drop=True)
        
        # Calculate p-values
        p_value = chi2.sf(X2, dff.values)
        
        # Get variable names
        if self.with_formula:
            if by_var:
                var_names = model.model.design_info.term_names
            else:
                var_names = model.model.design_info.column_names
        else:
            var_names = self.var_names
        
        v_names = ['Omnibus'] + var_names
        
        # Create results DataFrame with better column names
        results = pd.concat([
            pd.DataFrame({'Variable': v_names}),
            pd.DataFrame({'Chi-square': X2.reshape((-1,))}).round(3),
            dff,
            pd.DataFrame({'p-value': p_value.reshape((-1,))}).round(3)
        ], axis=1)
        

        return results

    def summary(self):
        """Generate a formatted summary of the Brant test results.
        
        Returns
        -------
        str
            A formatted string containing test results and interpretation
        """
        # Get results DataFrame
        results_df = self.TestResults()
        
        # Format header section
        model_type = "Formula-based" if self.with_formula else "Array-based"
        header = [
            "\nBrant Test of Parallel Regression Assumption",
            "=" * 50,
            f"Model type: {model_type}",
            f"Number of observations: {self.model.nobs}",
            f"Number of predictors: {len(self.var_names)}",
            f"Number of outcome levels: {len(np.unique(self.model.endog))}",
            "-" * 50,
            "\nTest Results:\n"
        ]
        
        # Format DataFrame as string with proper formatting
        results_str = results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:>12.3f}" if isinstance(x, (float, np.floating)) else f"{x:>12}"
        )
        
        # Add interpretation section
        interpretation = [
            "",
            "-" * 50,
            "\nInterpretation:",
            "* H0: Parallel Regression Assumption holds",
            "* P-value < 0.05 suggests violation of the parallel regression assumption",
            "* Omnibus test examines global validity of the assumption",
            "* Individual tests identify specific problematic variables",
            ""
        ]
        
        # Combine all sections
        formatted_output = "\n".join(header + [results_str] + interpretation)
        
        return formatted_output