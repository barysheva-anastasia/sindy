#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
Support functions for the sindy toolkit.

For method details, please see "A toolkit for data-driven discovery of governing equations in 
high-noise regimes" (2022) by C.B. Delahunt and J.N. Kutz.
         
Copyright (c) 2021 Charles B. Delahunt.  delahunt@uw.edu
MIT License
"""


import sys
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LinearRegression 
from numpy.linalg import lstsq  # since this accepts complex inputs and targets.
from scipy.linalg import lstsq as lstsqScipy  # the solver scikit-learn's LinearRegression uses.
from scipy.linalg import lu_factor as luFactor, lu_solve as luSolve  # trapping relaxation solve.
from scipy.linalg import schur as schurDecomposition, solve_triangular as solveTriangular
from scipy.linalg import solve_continuous_lyapunov as solveContinuousLyapunov

from scipy.integrate import solve_ivp

"""
---------------------------------------------------------------------------------------------
------------------------ Function Defs ------------------------------------------------------
--------------------------------------------------------------------------------------------- """

#%% Functions for various model systems, that return initial conditions for derivatives. 
# These are used to simulate the system via odeint():
 
def lorenz_fn(z, t, p=({'p0': 10, 'p1': 28, 'p2': np.round(-8/3, 2), 'numExtraVars': 0},)):
    """ xDot = -p0*x + p0*y
        yDot =  p1*x - y - x*z
        zDot =  p2*z + x*y  
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
            """  
    derivList = [
        p['p0'] * (z[1] - z[0]),
        z[0] * (p['p1'] - z[2]) - z[1],
        z[0] * z[1] + p['p2'] * z[2]
] 
    for i in range(p['numExtraVars']):
        derivList.append(0) 
    
    return derivList 

# End of lorenz attractor fn 
    
# -------------------------------------------------------------------    
def dampedHarmonicOscillatorLinear_fn(z, t, p=({'p0': 0.1, 'p1': 2, 'numExtraVars': 0})):
    """ xDot = -p0*x + p1*y
        yDot = -p1*x - p0*y  
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
    """  
    derivList = [
        -p['p0']*z[0] + p['p1']*z[1],
        -p['p1']*z[0] - p['p0']*z[1]
]
    for i in range(p['numExtraVars']):
        derivList.append(0) 
    
    return derivList 

# end of dampedHarmonicOscillatorLinear_fn
# -------------------------------------------------------------------
    
def dampedHarmonicOscillatorCubic_fn(z, t, p=({'p0': 0.1, 'p1': 2, 'numExtraVars': 0})):
    """ xDot = -p0*x^3 + p1*y^3
        yDot = -p1*x^3 - p0*y^3  
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
    """  
    derivList = [
        -p['p0']*pow(z[0], 3) + p['p1']*pow(z[1], 3), 
        -p['p1']*pow(z[0], 3) - p['p0']*pow(z[1], 3)
]
    for i in range(p['numExtraVars']):
        derivList.append(0) 
    
    return derivList 

# end of dampedHarmonicOscillatorCubic_fn
#---------------------------------------------------------------------- 
        
def threeDimLinear_fn(z, t, p=({'p0': 0.1, 'p1': 2, 'p2': 0.3, 'numExtraVars': 0})):
    """ xDot = -p0*x - p1*y
        yDot =  p1*x - p0*y 
        zDot = -p2*z
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
            
    NOTE: >= 4th order library terms cause failure in Brunton
    """  
    derivList = [
        -p['p0']*z[0] - p['p1']*z[1],
        p['p1']*z[0] - p['p0']*z[1],
        -p['p2']*z[2]
]
    for i in range(p['numExtraVars']):
        derivList.append(0) 
    
    return derivList 

# end of threeDimLinear_fn
# -------------------------------------------------------------------
     
def hopfNormalForm2D_fn(z, t, p=({'p0': 0, 'p1': -1, 'p2': 1, 'numExtraVars': 0})):
    """ Mean field model with zDot == 0:
        xDot = p0*x + p1*y - p2*x*(x^2 + y^2)
        yDot =  -p1*x + p0*y - p2*y*(x^2 + y^2)
        
        where p0 = mu, p1 = omega, p2 = A, p3 = lambda in Brunton paper. 
        
        Note that in the 3D model, zDot = -p2 * (z - x^2 - y^2). In this 2D version we assume 
        lambda is big, so zDot -> 0 rapidly and thus z = x^2 + y^2.
        TO-DO: we need param values for this model. mu is in [-0.2, 0.6]. omega, A, and lambda 
               values are unknown.
        Initial values estimate: x,y = {1, 0.75} or {0,0} (see fig 3 in Brunton paper)
        
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
            """  
    derivList = [
        p['p0'] * z[0] - p['p1'] * z[1] + p['p2'] * z[0] * (pow(z[0], 2) + pow(z[1], 2)),
        p['p1'] * z[0] + p['p0'] * z[1] + p['p2'] * z[1] * (pow(z[0], 2) + pow(z[1], 2))  
] 
    for i in range(p['numExtraVars']):
        derivList.append(0)
    
    return derivList 
 
# end of hopfNormalForm2D_fn
# -------------------------------------------------------------------
         
def hopfNormalForm3D_fn(z, t, p=({'p0': 0, 'p1': -1, 'p2': 1, 'p3': 0.5, 'numExtraVars': 0})):
    """ Mean field model with zDot == 0:
        xDot = p0*x + p1*y - p2*x*z 
        yDot =  -p1*x +p0*y - p2*y*z
        zDot = -p3 * (z - x^2 - y^2).
        
        where p0 = mu, p1 = omega, p2 = A, p3 = lambda in Brunton paper.
        In this 3D version of the model, we assume lambda is not too big, so zDot is not == 0.
        
        TO-DO: We need param values for this model. mu is in [-0.2, 0.6]. omega, A, and lambda 
        values are unknown. See tables 10, 11 in Brunton paper S.I. 
        Question: is mu being used in two ways, as a coefficient and as a "bifurcation parameter" 
                  (see eg table 13)?
        Initial values estimate: x,y = {1, 0.75} or {0,0} (see fig 3 in Brunton paper)
        
        Inputs: 
            z: np.vector of floats (initial conditions)
            t: np.vector of floats (timesteps)
            p: dict (system parameters)
        Output:
            derivList: np.vector of floats (initial conditions of derivatives).
            """  
    derivList = [
        p['p0'] * z[0] - p['p1'] * z[1] + p['p2'] * z[0] * z[2], # (pow(z[0], 2) + pow(z[1], 2)),
        p['p1'] * z[0] + p['p0'] * z[1] + p['p2'] * z[1] * z[2], # (pow(z[0], 2) + pow(z[1], 2)),
        -p['p3'] * (z[2] - pow(z[0],2) - pow(z[1], 2))]
    
    for i in range(p['numExtraVars']):
        derivList.append(0)
    
    return derivList 

# end of hopfNormalForm3D_fn
# ------------------------------------------------------------------

def generateModelStrAndTrueArrays_fn(modelSystem, p):
    """
    Build a string describing the model. 
    
    Parameters
    ----------
    modelSystem : str
    p : dict

    Returns
    -------
    modelStr : str.
    trueLib : tuple of lists of str
    trueLibCoeffs : tuple of lists of floats

    """
    if modelSystem == 'lorenz':   
        modelStr = \
        "x' = -" + str(p['p0']) + ' x + ' + str(p['p0']) + ' y' + '\n' + \
        "y' = " + str(p['p1']) + ' x - y - x*z' + '\n' + \
        "z' = " + str(p['p2']) + ' z' + ' + x*y' 
        trueLib = (['x','y'], ['x','y','x*z'], ['z', 'x*y'])
        trueLibCoeffs = ([-p['p0'], p['p0']], [p['p1'], -1, -1], [p['p2'], 1])
    if modelSystem == 'harmOscLinear':  
        modelStr = \
        "x' = -" + str(p['p0']) + ' x + ' + str(p['p1']) + ' y' + '\n' + \
        "y' = -" + str(p['p1']) + " x -" + str(p['p0']) + ' y' 
        trueLib = (['x', 'y'], ['x', 'y'])
        trueLibCoeffs = ([-p['p0'], p['p1']], [-p['p1'], -p['p0']])
    if modelSystem == 'harmOscCubic':  
        modelStr = \
        "x' = -" + str(p['p0']) + ' x^3 + ' + str(p['p1']) + ' y^3' + '\n' + \
        "y' = -" + str(p['p1']) + " x^3 -" + str(p['p0']) + ' y^3' 
        trueLib = (['x^3', 'y^3'], ['x^3', 'y^3'])
        trueLibCoeffs = ([-p['p0'], p['p1']], [-p['p1'], -p['p0']])
    if modelSystem == 'threeDimLinear':        
        modelStr = \
        "x' = -" + str(p['p0']) + ' x - ' + str(p['p1']) + ' y' + '\n' + \
        "y' = " + str(p['p1']) + " x -" + str(p['p0']) + ' y' + '\n' + \
        "z' = -" + str(p['p2']) + " z" 
        trueLib = (['x', 'y'], ['x', 'y'], ['z'])
        trueLibCoeffs = ([-p['p0'], -p['p1']], [p['p1'], -p['p0']], [-p['p2']])
    if modelSystem == 'hopfNormal2D':         
        modelStr = \
        "x' = " + str(p['p0']) + ' x + ' + str(p['p1']) + ' y ' + "+ " + str(p['p2']) + \
                '(x^3 + x*y^2)' + '\n' + \
        "y' = " + str(p['p1']) + ' x + ' + str(p['p0']) + ' y ' + "+ " + str(p['p2']) + \
                '(y*x^2 + y^3)' 
        trueLib = (['x', 'y', 'x^3', 'x*y^2'], ['x', 'y', 'y^3', 'x^2*y'])
        trueLibCoeffs = ([p['p0'], p['p1'], p['p2'], p['p2']], 
                         [p['p1'], p['p0'], p['p2'], p['p2']])
    if modelSystem == 'hopfNormal3D':        
        modelStr = \
        "x' = " + str(p['p0']) + ' x - ' + str(p['p1']) + ' y ' + "+ " + str(p['p2']) + \
                ' x*z' + '\n' + \
        "y' = " + str(p['p1']) + ' x + ' + str(p['p0']) + ' y ' + "+ " + str(p['p2']) + \
                ' y*z' + '\n' + \
        "z' = -" + str(p['p3']) + ' * (z - x^2 - y^2)' 
        trueLib = (['x', 'y', 'x*z'], ['x', 'y', 'y*z'], ['z', 'x^2', 'y^2'])
        trueLibCoeffs = ([p['p0'], p['p1'], p['p2']], 
                         [p['p1'], p['p0'], p['p2']], [-p['p3'], p['p3'], p['p3']])
        
    return modelStr, trueLib, trueLibCoeffs

# End of generateModelStrAndTrueArrays_fn 
# ---------------------------------------------------

def generateTrueFunctionalAndCoeffArrays_fn(trueLib, trueLibCoeffs, functionList):
    """
    Given lists of functional names as str, and the function list, construct a true 
    'functionsToUseArray'

    Parameters
    ----------
    trueLib : list-like of lists of str. len of list-like = numVars, len of each list = num true
              library functionals for that variable
    trueLibCoeffs : list-like of lists of floats. Matches 'trueLib' above.
    functionList : list of str.  The functional names

    Returns
    -------
    trueLibraryArray : np.array of bools, numVars x numFunctionals

    """
    trueLibraryArray = np.zeros((len(trueLib), len(functionList)), dtype=bool)
    trueCoeffArray = np.zeros((len(trueLib), len(functionList)))
    for v in range(len(trueLib)):  # v is the variable index
        theseFnalNames = np.array(trueLib[v])
        theseCoeffs = np.array(trueLibCoeffs[v])
        for f in range(len(functionList)):
            ind = np.where(theseFnalNames == functionList[f])[0]
            if len(ind) > 0:  # ie functionList[f] is a true functional
                trueLibraryArray[v, f] = True  
                trueCoeffArray[v, f] = theseCoeffs[ind[0]]
    
    return trueLibraryArray, trueCoeffArray

# End of generateTrueFunctionalAndCoeffArrays_fn
# ----------------------------------------------------------
    
def generateFunctionStr_fn(v, varInds, variableNames):
    """
    Given a list, generate a string. Used by generatePolynomialLibrary_fn.

    Parameters
    ----------
    v : list of ints
    varInds : list of ints
    variableNames : list of str

    Returns
    -------
    fnStr : str

    """
    fnStr = ''
    for i in varInds:
        if i in v:
            if len(fnStr) > 0:  # case: we need a multiplication sign:
                fnStr += '*'
            fnStr += variableNames[i]
            if np.sum(np.array(v) == i) > 1: # case: we need an exponent:
                fnStr += '^' + str(np.sum(np.array(v) == i))
    return fnStr
# End of generateFunctionStr_fn
#------------------------------------------------------------------------
 
def generatePolynomialLibrary_fn(variableNames, degree):
    """
    Generate a library of polynomials up to a certain degree. Return two things: a list of 
    functional names and a list of recipes for use in ode evolutions.
    NOTE: If there is one variable, and its name is more that one character, this function will 
    fail because 'varInds' will equal the number of characters in the variable name.

    Parameters
    ----------
    variableNames : list-like of str 
    degree : int

    Returns
    -------
    functionList : list of str 
    recipes : list of lists, length = numFunctionals
    """

    varInds = np.arange(len(variableNames))
    recipes = []
    functionList = []
    recipes.append(-1)   # the constant function
    functionList.append('1') 
    
    # Treat degree = 1:
    combos = []  # initialize  
    for i in varInds:
        combos.append(i)    
        recipes.append(i)
        functionList.append(variableNames[i])
        
    deg = 2  # initialize 
    while deg <= degree: 
        combos = [(i, j) for i in varInds for j in combos]  # vector of {int, list} entries, 
        # entry has total len = deg. There are duplicates at >= degree 3, eg (0,(0,1)) and 
        # (1,(0,0)). So for each entry we must (a) make into a single list; and (b) check that it
        # is new before appending it to 'recipes' and 'functionList'.
        # (a) combine int and list:
        keepCombos = []  # to keep the non-duplicates
        for k in range(len(combos)): 
            c = combos[k]    
            this = []   
            for i in c:
                if isinstance(i, (int, np.integer)):
                    this.append(i)
                else:
                    for j in i:
                        this.append(j)
            this = list(np.sort(np.array(this)))
            # 'this' is now a single sorted list of ints.
            # (b) If 'this' is new, append to recipes:
            addFlag = True
            for i in recipes:
                if not isinstance(i, (int, np.integer)):
                    if len(this) == len(i) and np.sum(np.array(this) == np.array(i)) == len(i):
                        addFlag = False
                        break
            if addFlag:
                recipes.append(this)  
                keepCombos.append(this)  
                functionList.append(generateFunctionStr_fn(this, varInds, variableNames))  
                    
        # Update combos with non-duplicate list:
        combos = keepCombos
        deg += 1
    
    return functionList, recipes

# End of generatePolynomialLibrary_fn
#--------------------------------------------------------------

def calculateLibraryFunctionalValues_fn(x, recipes):
    """
    For each functional, calculate its values at the timepoints.

    Parameters
    ----------
    x : np.array of floats, numTimepoints x numVars
    recipes : list of lists. The i'th list gives the indices of variables to be multiplied together 
              to generate the i'th functional.

    Returns
    -------
    fnVals : np.array of floats, numTimepoints x numFunctionals.
    """
    
    fnVals = np.zeros((x.shape[0], len(recipes)))
    for i in range(fnVals.shape[1]):
        r = recipes[i]
        temp = np.ones(x.shape[0])
        if isinstance(r, (int, np.integer)):  # constant or degree 1 monomial
            if r != -1:  # ie not the constant functional 
                temp = x[:, r]
        else:  # >= degree 2
            for j in r:
                temp = temp * x[:, j]
        fnVals[:, i] = temp
        
    return fnVals

# End of calculateLibraryFunctionalValues_fn
#------------------------------------------------------------

#%% Make a hamming window:
def makeHammingWindow_fn(hammingWindowLength, plateauRatio=0):
    """"  Generate a hamming window, perhaps with a flat plateau in the middle (ie a smoothed step
          function).
    Inputs:
        hammingWindowLength: int
        usePlateauHammingFilterFlag: Boolean
        plateauRatio: float 0 to 1
    Outputs:
        hamm: vector with sum = 1
        """
    if plateauRatio > 0:
        # add a plateau in the middle:
        plateauRatio = min(1, plateauRatio)
        ends = int(np.ceil(hammingWindowLength*(1 - plateauRatio)))
        if ends%2 == 1:
            ends = ends + 1   # make even
        rise = int(ends/2)
        ends = np.hamming(ends)    # ends is now a hamming vector
        hamm = np.ones((1, hammingWindowLength))
        hamm = hamm.flatten()
        hamm[0:rise] = ends[0:rise]
        hamm[-rise:] = ends[-rise:]
    else:
        # normal hamming filter
        hamm = np.hamming(hammingWindowLength)
    # Normalize:
    hamm = hamm / np.sum(hamm)

    return hamm

# End of makeHammingWindow_fn
#---------------------------------------------------------

def calculateSlopeAndStd_fn(x, dt, w):
    ''' given a time-series, do two things:
        1. calculate the deriv at each point by simple rise/run (Euler formula)
        2. calculate the std of a window (size2*h) at each point, using the slope from (1) to
           first tilt the data to roughly slope = 1.
    Inputs:
        z: np.vector
        dt: float
        w: int. Window length
    Outputs:
        slopeX: np.vector
        stdX: np.vector
        meanX: np.vector
        '''
    h = int(np.round(w/2))  # Half the window length
    slopeX = np.zeros(x.shape)
    stdX = np.zeros(x.shape)
    meanX = np.zeros(x.shape)
    # For efficiency, we take the mean of the first window, then update it at each new point:
    for i in range(len(x)):
        if i == h + 1: # First point's window
            b = np.mean(x[i:i + h])
            a = np.mean(x[i-h:i])
        if i > h + 1 and i < len(x) - h:  # all ensuing points' windows
            b = b + (x[i + h] - x[i])/h
            a = a + (x[i] - x[i-h])/h
        if i > h and i < len(x) - h:  # all points' windows (ie happens for both above cases)
            slopeX[i] = (b-a)/h
            tilted = x[i-h:i+h] - slopeX[i]*np.array(range(-h, h))
            stdX[i] = np.std(tilted)  # subsampling doesn't speed this up much.
            meanX[i] = 0.5 * (b + a)

    # Fill in start and end values:
    slopeX[0:h + 1] = slopeX[h + 1]
    slopeX[-h:] = slopeX[-h - 1]
    stdX[0:h + 1] = stdX[h + 1]
    stdX[-h:] = stdX[-h - 1]
    meanX[0:h + 1] = meanX[h + 1]
    meanX[-h:] = meanX[-h - 1]

    # account for dt:
    slopeX = slopeX/dt

    return slopeX, stdX, meanX

# End of calculateSlopeAndStd_fn

#------------------------------------------------------------------

def addGaussianNoise_fn(X, noiseFactors, noiseType = 'normal'):
    """ add noise (default gaussian) directly to the trajectory array.
    Inputs:
        X: n x m np.array. n = number of variables, m = number of timepoints, ie each row is
        variable time-course
        noiseFactors: np.array of floats, num real vars x 1
        noiseType: str,  'normal' or 'uniform'
        extraVarsnoiseFactors: float
    Outputs:
        XwithNoise: n x m np.array
        """

    noiseArray = np.tile(noiseFactors, [X.shape[0], 1])
    # Add  noise to xTrain, scaled as fraction of std dev of x, y, z values over the trajectory:
    scalingVals  = np.std(X, axis = 0)  # for scaling noise. Use only x, y, z
    # 'scalingVals' for extra variables currently == 0, so replace these with the mean of others:
    scalingVals[scalingVals < 1e-5] = np.mean(scalingVals[scalingVals > 1e-5])

    # Add rows for the extra variables:
    if noiseType == 'uniform':
        noiseToAdd = 2 * noiseArray * np.multiply(np.tile(scalingVals, (X.shape[0], 1)),
                                                 -0.5 + np.random.uniform(size = X.shape))
    else: #  noiseType == 'normal':
        noiseToAdd = noiseArray * np.multiply(np.tile(scalingVals, (X.shape[0], 1)),
                                              np.random.normal(size = X.shape))
    xNoisy = X + noiseToAdd

    return xNoisy

# End of addNoise_fn
#-----------------------------------------------------------------------------------------

def addWhiteNoise_fn(xArray, noiseFactors):
    ''' Add white noise to trajectory array, using FFT.
    Inputs:
        xArray: np.array, numTimepoints x numVars
        noiseFactors:  np.array of floats, num real vars x 1
    Outputs:
        xNoisy: np.array, real part of ifft of noisy fft signal.
        '''

    xNoisy = np.zeros(xArray.shape)
    for i in range(xArray.shape[1]):
        x = xArray[:, i]
        xF = np.fft.fft(x)
        realSigma = np.std(np.real(xF))
        imSigma = np.std(np.imag(xF))
        noiseToAdd = noiseFactors[i] * (realSigma * np.random.normal(size=xF.shape) + \
                                        1j * imSigma * np.random.normal(size=xF.shape))
        xFNoisy = xF + noiseToAdd
        xNoisy[:, i] = np.fft.ifft(xFNoisy)
    return np.real(xNoisy)

# End of addWhiteNoise_fn
#--------------------------------------------------------------------------------------

def calcWeightArray_fn(cNew, functionsToUseArray, imputed, coeffWeightsCutoffFactor, 
                       percentileOfImputedValuesForWeights):
    """
    Create a weight array to weight the coeffs by size of functional values. Culling will be
    based on the weighted coeffs.
    We give high weights to functionals that tend to have high values, since we want to allow
    them to have lower coeffs without being culled. Functionals that tend to have low values
    get low weights, since their effect will be less. We 'normalize' using the median of
    all functional imputed values.

    Parameters
    ----------
    cNew : np.array of floats. numVars x numFunctionals.
    functionsToUseArray : np.array of booleans. numVars x numFunctionals.
    imputed : np.array of floats. vector 1 x numFunctionals. Estimated magnitudes of each
              functional. 
    coeffWeightsCutoffFactor : (scalar float or int)
    percentileOfImputedValuesForWeights : scalar int

    Returns
    -------
    weightArray : np.array of floats. numVars x numFunctions.

    """
    # Reduce the extremes of this vector in each row (ie for each variable), and normalize:
    weightArray= np.zeros(cNew.shape)
    for i in range(weightArray.shape[0]):
        if np.sum(functionsToUseArray[i, :]) > 0:
            temp = imputed.copy()
            centralVal = np.percentile(temp[functionsToUseArray[i, :]],
                                       percentileOfImputedValuesForWeights, 
                                       method='lower')  
            # Over functionals currently active for this variable.
            # Moderate the extremes of this vector:
            temp[temp >  coeffWeightsCutoffFactor * centralVal] = \
                 coeffWeightsCutoffFactor * centralVal
            temp[temp < centralVal / coeffWeightsCutoffFactor] = \
                centralVal / coeffWeightsCutoffFactor
            temp = temp * functionsToUseArray[i, :]  # Zero the weights of unused functions.
            # Normalize (comment: It would be nice if the normalization here penalized variables
            # with many active functionals, to enforce sparsity.):
            temp = temp / np.median(temp[temp > 0])  # So in the ballpark of 1
            weightArray[i, :] = temp

    return weightArray
#-------- End of calcWeightArray_fn-----------------

def calculateFftForRegression_fn(x, numFftPoints, fftRegressionTarget):
    """
    Create a vector using some form of FFT, to act as a regression target. 

    Parameters
    ----------
    x : np.array (vector) of floats
    numFftPoints : int
    fftRegressionTarget : str

    Returns
    -------
    xT : np.array (vector) of floats

    """
    x = x - np.mean(x)
    xP = np.fft.fft(x)  # Use this if fftRegressionTarget == 'complex'
    if fftRegressionTarget == 'realOnly':
        xP = np.real(xP)
    if fftRegressionTarget == 'magnitude':
        xP = np.abs(xP)
    if fftRegressionTarget == 'power':
        xP = pow(np.real(xP), 2)
        
    xP = xP[0:numFftPoints] / np.sum(np.abs(xP[0:numFftPoints]))  # normalize
    
    return xP
# ------- End of calculateFftRegressionTarget_fn ----------------------------

def cullAndAssessWhetherToRerun_fn(localCoeffArray, variableNames,  functionList,
                                   imputedSizeOfFunctionals, cullingRulesDict, functionsToUseArray,
                                   cullable, inBalanceVarsArray, coeffWeightsCutoffFactor,
                                   percentileOfImputedValuesForWeights, uncullableFnsArray=None):
    """
    Given results of post-smoothing sindy or linear fit, see if we can cull any variables or
    library functions.
    We make an array of booleans that shows which coeffs of the model were non-zero. This array
    serves as modelActiveLib, restricting which library functions can be used for each of
    the variables. No variables or functions are actually removed. Instead, they are set off-limits
    by modifying the boolean array.

    Parameters
    ----------
    localCoeffArray : np.array of floats, numVars x numFunctionals
    variableNames : vector of strings
    functionList : vector of strings
    imputedSizeOfFunctionals : np.array of floats
    cullingRulesDict : Dict with keys like 'maxAllowedWeightedCoeff'
    functionsToUseArray : np.array booleans, numVars x numFunctionals
    cullable : np.array booleans, numVars x numFunctionals
    inBalanceVarsArray :  np.array booleans, numVars x numFunctionals. True = eligible for culling
    coeffWeightsCutoffFactor : scalar float
    percentileOfImputedValuesForWeights : int
    uncullableFnsArray : np.array of booleans, numVars x numFunctionals, or None. Functionals that
                         must not be culled at all, whatever their weighted coeff. Unlike 'cullable'
                         this is also honored by the too-small cull in step 4 below, which is the
                         point of it: 'cullable' carries the restore-and-protect mechanism's
                         book-keeping, and making step 4 consult that would change every existing
                         run. The caller uses this for functionals that the boundedness constraint
                         restored, which the too-small cull would otherwise remove again on the next
                         iteration, forever. None -> nothing is exempt, ie the historical behavior.

    Returns
    -------
    iterateAgainFlag : Boolean
    functionsToUseArray : np.array of booleans, numVars x numFunctions
    outputStr : str
    minNonZeroVal : float, the minimum weighted coeff (to use as a stopping signal)
    """
 
    numExtraToKill = cullingRulesDict['extraNumFnsCulledPerIter']
    minAllowedWeightedCoeff = cullingRulesDict['minAllowedWeightedCoeff']
    # Note: these are trivially == 1 and have no effect if 'coeffWeightsCutoffFactor' == 1:

    # Exempt the functionals listed in 'uncullableFnsArray' from both culls below. Doing it here
    # covers the ratcheting-threshold cull in step 7, which already consults 'cullable'; step 4 has
    # its own conjunction, since it ignores 'cullable' by design:
    if uncullableFnsArray is None:
        uncullableFnsArray = np.zeros(functionsToUseArray.shape, dtype=bool)
    cullable = np.logical_and(cullable, np.logical_not(uncullableFnsArray))


    cOld = localCoeffArray  # i'th col corresponds to the i'th entry in functionList. j'th
    #row corresponds to j'th variable.
    cNew = cOld.copy()           # we'll zero out entries in this array
    fOld = functionsToUseArray.copy()  # fOld is for comparison, to determine 'iterateAgainFlag', 
    # since we'll update functionsToUseArray.    
    # We could also get fOld from modelActiveLib

    # Kill the constant term if it is the only non-zero term:
    if cullingRulesDict['setConstantDerivEstimatesToZero']:
        for i in range(cNew.shape[0]):
            if np.sum(np.abs(cNew[i, 1:])) < 1e-5:  # 1e-5 = almostZeroRoundDownThreshold, which
            # deals with weird not-quite-zero zeros
                cNew[i,0] = 0

    # 1. find which vars are non-zero:
    zeroedVars =  np.sum(cNew, axis = 1) == 0 # col vector, True -> var has been zeroed out.

    # 2. Now zero out all functions that involve zeroed-out variables, ie zero out columns that
    # contain the var name:
    # Note: this has not been updated to use 'cullable'.
    for i in range(len(zeroedVars)):
        if zeroedVars[i] == True:
            varName = variableNames[i]
            for j in range(cNew.shape[1]):
                if varName in functionList[j]:
                    cNew[:,j] = 0

    # 3. Create a weight array to weight the coeffs by size of functional values.
    functionsToUseArray = np.abs(cNew) > 1e-5  # almostZeroRoundDownThreshold
    if coeffWeightsCutoffFactor > 1:  # Recall coeffWeightsCutoffFactor = 1 -> no weights.
        weightArray = calcWeightArray_fn(cNew, functionsToUseArray, imputedSizeOfFunctionals, 
                                         coeffWeightsCutoffFactor, 
                                         percentileOfImputedValuesForWeights)
    else:  # Case: we're not weighting the coeffs.
        weightArray = np.ones(cNew.shape) / cNew.shape[1]
    # weightArray is now ready to multiply cNew.

    # 3. Now zero out all functions with coeffs that are too large. Currently never activates (at
    # maxAllowed = 50: we can disable this by picking a very large 'maxAllowedWeightedCoeff' 
    # parameter. inBalanceVarsArray == true prevents culling fnals from vars with low fnal counts.
    tooBigStr = ''
    # cNewWeighted = cNew * weightArray
    # tooBigFnsArray = np.logical_and(np.abs(cNewWeighted) > maxAllowedWeightedCoeff,
    #                                 inBalanceVarsArray)
    # make string of too-big functionals:
    # for i in np.where(np.sum(tooBigFnsArray,axis=1) > 0)[0]:  # ie rows/variables with too-small
    # # functionals.
    #     tooBigStr = tooBigStr + variableNames[i] + ': ' + \
    #         str(np.array(functionList)[tooBigFnsArray[i, :]]) + ' '
    # cNew[np.where(tooBigFnsArray)] = 0  # set too-big coeffs to 0

    # numTooBigCulledFunctionals = np.sum(tooBigFnsArray.flatten())

    # 4. Make an updated boolean array of retained functions. Also cull any functionals with very
    # low weighted coeffs:
    cNewWeighted = cNew * weightArray
    tooSmallFnsArray = np.logical_and.reduce((np.abs(cNewWeighted) < minAllowedWeightedCoeff,
                                      np.abs(cNewWeighted) > 0, inBalanceVarsArray,
                                      np.logical_not(uncullableFnsArray)))
    cNew[np.where(tooSmallFnsArray)] = 0  # set too-small coeffs to 0 
    # make string of too-small functionals:
    tooSmallStr = ''
    for i in np.where(np.sum(tooSmallFnsArray,axis=1) > 0)[0]:  # ie rows/variables with too-small
    # functionals.
        tooSmallStr = tooSmallStr + variableNames[i] + ': ' + \
            str(np.array(functionList)[tooSmallFnsArray[i, :]]) + ' '
    numTooSmallCulledFunctionals = np.sum(tooSmallFnsArray.flatten())

    # 5. see if a variable has been removed:
    oldZeroedVars = np.sum(cOld, axis = 1) == 0
    variableRemovedFlag =  np.sum(zeroedVars) > np.sum(oldZeroedVars)

    functionsToUseArray = np.abs(cNew) > 1e-5  # almostZeroRoundDownThreshold  # Update to remove 
    # too small coeffs
    
    # 6. See if an entire function (a column) has been removed:
    oldZeroedFns = np.sum(fOld, axis = 0)  # Sum the columns
    newZeroedFns = np.sum(functionsToUseArray, axis = 0)
    functionColumnRemovedFlag = np.sum(newZeroedFns == 0) - np.sum(oldZeroedFns == 0) > 0
    # 7. Cull based on weighted coeff size relative to current threshold:
    # If we have not removed a functional column, and if the lowest weighted coeff is small enough,
    # cull more coeffs. 
    # Make a new weight array, since we may have removed some functionals: 
    if coeffWeightsCutoffFactor > 1:  # Recall coeffWeightsCutoffFactor = 1 -> no weights.
        weightArray = calcWeightArray_fn(cNew, functionsToUseArray, imputedSizeOfFunctionals, 
                                         coeffWeightsCutoffFactor, 
                                         percentileOfImputedValuesForWeights)
    else:  # Case: we're not weighting the coeffs.
        weightArray = np.ones(cNew.shape) / cNew.shape[1]  
        
    cNewWeighted = cNew * weightArray
    
    # The guard must also require that at least one CULLABLE coeff is non-zero, as the same
    # calculation in step 8 below does. Checking only the total weight raises 'zero-size array to
    # reduction operation minimum' once every remaining non-zero coeff is protected, which happens
    # late in a run on a large library.
    if np.sum(np.abs(cNewWeighted.flatten())) > 1e-5 and \
        np.sum(np.logical_and(cullable, cNewWeighted != 0).flatten()) > 0:
        # 1e-5 = almostZeroRoundDownThreshold
        minNonZeroVal = np.min(np.abs(cNewWeighted)[np.logical_and(cullable,
                                                                   cNewWeighted != 0)].flatten())
    else:
        minNonZeroVal = 0
 
    extraFunctionalKilledOffFlag = False
    numExtraFunctionalsKilled = 0
    extraCullBypassedDueToRemovedFunctionColumnFlag = functionColumnRemovedFlag

    # If we removed a function from all vars (ie a column), we're done for this cull. Else see if
    # any functionals have coeffs below the current threshold. We ignore the protected fns (ie we
    # only consider 'cullable' fns)
    culledFnsArray = np.zeros(functionsToUseArray.shape, dtype=bool)  # to record functionals
    # culled because they were the most below the current threshold (but not including 'too-small'
    # fns).
    while numExtraToKill > 0 and (functionColumnRemovedFlag == False) and \
        minNonZeroVal > 0 and \
        np.sum(np.abs((cNewWeighted * inBalanceVarsArray).flatten())) > 1e-5:
        # 1e-5 = almostZeroRoundDownThreshold
        # 'minNonZeroVal > 0' means: there is a cullable non-zero coeff to cull. Without it the
        # loop would match the coeffs equal to zero and 'cull' functionals that are already gone.
        # It never changes a run where a cullable coeff exists, since a min over non-zero
        # magnitudes is positive.
        loc = np.where(np.logical_and.reduce((cullable, np.abs(cNewWeighted) == minNonZeroVal,
                                              inBalanceVarsArray)))
        functionsToUseArray[loc[0], loc[1]] = False  # Update the boolean functionals array
        cNewWeighted[loc[0], loc[1]] = 0  # Update the beta weights array
        culledFnsArray[loc[0], loc[1]] = True  # Record this culled functional
        numExtraFunctionalsKilled += len(loc[0])
        # outputStr = outputStr + '; ' + str(functionList[loc[1][0]]) + \
        #     ' (' + str(variableNames[loc[0][0]] + ')')
        extraFunctionalKilledOffFlag = True
        # Update 'functionRemovedFlag' accounting for newly-culled function:
        newZeroedFns = np.sum(functionsToUseArray > 0, axis = 0)
        functionColumnRemovedFlag = np.sum(newZeroedFns == 0) - np.sum(oldZeroedFns == 0) > 0
            
        temp = cNewWeighted * inBalanceVarsArray * cullable
        if np.sum(np.abs(temp.flatten())) > 1e-5:  # almostZeroRoundDownThreshold
            minNonZeroVal = np.min(np.abs(temp[temp != 0].flatten())) 
            if 'loc' in locals():
                numExtraToKill -= len(loc[0])
            else:
                numExtraToKill -= 1
        else:  # escape
            minNonZeroVal = 0
            numExtraToKill = 0
            
        

    culledStr = ''
    # List the culled fns in a str:
    for i in np.where(np.sum(culledFnsArray, axis=1) > 0)[0]:
        culledStr = culledStr + variableNames[i] + ': ' + \
            str(np.array(functionList)[culledFnsArray[i, :]]) + ' ' 

    # 8. If there are any functions still below thisThreshold, we want to rerun without changing 
    # the threshold, in order to pick them off one by one:
    if np.sum(np.abs(cNewWeighted.flatten())) > 1e-5 and \
        np.sum(np.logical_and(cullable, cNewWeighted != 0).flatten()) > 0:
        newMinNonZeroVal = np.min(np.abs(cNewWeighted)[np.logical_and(cullable,
                                                                  cNewWeighted != 0)].flatten())    
        cullableFunctionFlag = True
    else:
        cullableFunctionFlag = False

    # Make a combined too-big, too-small output str:
    # if numTooBigCulledFunctionals > 0:
    #     tooBigStr = '. Culled ' + tooBigStr + ' with weighted coeffs > ' + \
    #         str(maxAllowedWeightedCoeff) + '. '
    if numTooSmallCulledFunctionals > 0:
        tooSmallStr = 'Culled ' + tooSmallStr + \
            ' with weighted coeffs < ' + str(minAllowedWeightedCoeff) + '. '
    if numExtraFunctionalsKilled > 0:
        culledStr = 'Culled ' + culledStr
    else:
        culledStr = ''
        
    if extraCullBypassedDueToRemovedFunctionColumnFlag:
        culledStr = ' Extra culling step bypassed due to removed function column.'

    outputStr = tooBigStr + tooSmallStr + culledStr
    
    # if any new variable, new function, or even one new functional has been removed, we will do
    # another cull.
    iterateAgainFlag = variableRemovedFlag or functionColumnRemovedFlag or \
        extraFunctionalKilledOffFlag or cullableFunctionFlag

    # Note: the returned arg 'culledFnsArray' shows the functionals culled in a way (by being 
    # below the ratcheting threshold) that makes them eligible to be restored. Functionals that 
    # have too-big or too-small weighted coeffs are not restorable.

    return iterateAgainFlag, functionsToUseArray, culledFnsArray, outputStr, \
        minNonZeroVal, cullableFunctionFlag

# End of cullAndAssessWhetherToRerun_fn 
# -----------------------------------------------------------------------------------------
  

def smoothData_fn(x, window, smoothType='hamming'):
    """
    Smooth each column of an input array using a window.
    
    Parameters
    ----------
    x : np.array of floats, likely numTimepoints x numSomething (variables or functionals)
    window : np.array, vector of floats
    smoothType : str, eg 'hamming' 

    Returns
    -------
    xSmoothed : np.array, size x.shape

    """
    xSmoothed = np.zeros(x.shape)
    if True:  # smoothType == 'hamming': Assume it's always hamming
        for i in range(x.shape[1]):
            temp = x[:, i] 
            xSmoothed[:, i] = np.convolve(temp - temp[0], window, mode='same') + temp[0]
            # shift time-series to start at 0 for the convolution, to mitigate edge effects

    return xSmoothed
#  End of smoothLibraryFunctionsForRegression_fn
#-----------------------------------------------------------------------

def parseTimepointsByMagnitudesOfVariables_fn(x, functionsArray, nonConstantIndices, margin, 
                                              maxFnValRatio, minNumStartIndsToUse, 
                                              removeMarginFlag=True):
    """
    Accept or reject timepoints to use in regressions, based on whether the functionals have
    widely disparate values or not.

    Parameters
    ----------
    x : np.array of floats, numTimepoints x numFunctionals. Each column is time-series of a
        functional's values.
    functionsArray : np.array of booleans, numVariables x numFunctionals.
    nonConstantIndices : list of ints (generated after the functional library is created)
    margin : int
    maxFnValRatio : float
    minNumStartIndsToUse : int
    removeMarginFlag : bool
  
    Returns
    -------
    startIndsToUse : np.array of booleans numVariables x numTimepoints (not counting margins at
                     either end).

    """
    lengthOfPadToIgnore = 0
    if removeMarginFlag:
        lengthOfPadToIgnore = margin
    startIndsToUse = np.zeros((functionsArray.shape[0], x.shape[0] - 2*lengthOfPadToIgnore),
                              dtype=bool)

    for var in range(functionsArray.shape[0]):
        if np.sum(functionsArray[var,]) == 0:  # Ignore fully culled variables
            pass
        else:
            thisX = x[margin:-margin,] if removeMarginFlag else x  # Ignore outer margins. Subset
            # before multiplying (and broadcast rather than tile) to save a full-size temporary.
            if not np.all(functionsArray[var,] == 1):
                thisX = thisX * functionsArray[var,]
                # When every entry of the mask is exactly 1 the multiply is an identity, so it is
                # skipped. 'thisX' is only read below, so aliasing x in that case is safe.
            # At spaced timepoints, find the median of abs() of non-zero values, and use
            # that as an anchor for scaling:
            subsampleRate = 3
            # Vectorized over the subsampled timepoints: the original loop evaluated one row per
            # 'subsampleRate'-sized block and copied the result across the whole block, so we
            # evaluate all block rows at once and then repeat. nanmedian over the zeros-blanked
            # rows reproduces median(v[v != 0]) row by row, including the all-zeros -> nan case.
            v = np.abs(thisX[0::subsampleRate,])
            m = np.nanmedian(np.where(v != 0, v, np.nan), axis=1).reshape(-1, 1)
            okToUseByBlock = \
                np.logical_and(np.logical_and(v < m * maxFnValRatio, v > m / maxFnValRatio),
                               functionsArray[var,])  # chained rather than reduce() over a tuple,
                # since the row mask now broadcasts against 2-D operands.
            okToUse = np.repeat(okToUseByBlock, subsampleRate, axis=0)[0:thisX.shape[0],]
            # 'okToUse' is a boolean array that tells us, for each timepoint, which functionals 
            # are ok to regress on. We now select the timepoints that are valid for the highest 
            # number of functionals: 

            numOkFnsAtEachTimepoint = np.sum(okToUse, axis=1)
            # 'removeMarginFlag' = False is a signal that we are testing linear dependence, so we
            # wish to use all the columns of functionsArray because we have already subselected
            # the functionals of interest.
            # Note: a 'counts' histogram of numOkFnsAtEachTimepoint used to be built here. It fed
            # only the commented-out diagnostic printout below, so it has been dropped.

            # Keep only those timepoints that use all non-constant available fns
            # (captured in 'countValToUse'), ie ignore any timepoints where any
            # two functions exceed the maxFnValRatio-derived bounds, EXCEPT subject to a 
            # guaranteed minimum number of timepoints 'minNumStartIndsToUse', enforced in the 
            # while loop below.
            if removeMarginFlag:
                countValToUse = np.sum(functionsArray[var, nonConstantIndices])
            else:  # case: lin dependence context, use all columns of functionsArray
                countValToUse = np.sum(functionsArray[var, :])
            # fullCountValToUse = countValToUse.copy()  # used for diagnostic printout below.
            startIndsToUse[var,] = numOkFnsAtEachTimepoint >= countValToUse
            # Add a catch to prevent too few startInds:
            while np.sum(startIndsToUse[var,]) < minNumStartIndsToUse:
                countValToUse = countValToUse - 1
                startIndsToUse[var,] = numOkFnsAtEachTimepoint >= countValToUse
            # Diagnostic printount:
            # print('var ' + str(var) + ': functional count used for selecting timepoints: ' + \
            #       str(countValToUse) + ' out of ' + str(fullCountValToUse) + \
            #       ', numTimepoints = ' + str(np.sum(startIndsToUse[var,])) + ' out of ' + \
            #       str(startIndsToUse.shape[1])) 
                
    # 'startIndsToUse' is an array of booleans, which say whether a certain
    # timepoint (starting at 'margin' and ending at {len(tTrain) - margin}) is eligible
    # to use.

    return startIndsToUse

#  End of parseTimepointsByMagnitudesOfVariables_fn
# ---------------------------------------------------------------------------

def calculateDerivativesFromData_fn(x, t, method='centralDifference'):
    """
    Given time-series of variables x, calculate the derivatives of each variable. 
    This function differs from 'estimateDerivatives_fn' below by (a) accepting arrays (ie multiple
    time-series); (b) handling endpoints; (c) not returning weights. Could be combined.
    NOTE: Currently only allows central difference method

    Parameters
    ----------
    x: np.array of floats, numTimepoints x numVars 
    t : np.vector of floats. The timepoints

    Returns
    -------
    df : np.array of floats, numTimepoints x numFunctionals

    """
    t = t.reshape(-1,1)
    # if True method == 'centralDifference':
        
    dfMiddle = (x[2:, :] - x[0:-2, :]) / np.tile(t[2:] - t[0:-2], (1, x.shape[1]))
    dfFirst = (x[1, :] - x[0, :]) / (t[1] - t[0])
    dfLast = (x[-1, :] - x[-2, :]) / (t[-1] - t[-2])
    
    df = np.vstack((dfFirst, dfMiddle, dfLast))
        
    return df

# End of calculateDerivativesFromData_fn
#------------------------------------------------------------------

def calculateDerivativesFromModel_fn(x, coeffs, fVals):
    """
    Given a model with coefficients for functionals, calculate the derivatives based on this model,
    ie return what the model thinks the derivatives are.
    This is different from 'calculateDerivativesFromData_fn' above, which returns the central difference 
    derivative of the time-series.

    Parameters
    ----------
    x : np.array of floats, numTimepoints x numVars
    coeffs : np.array of floats, numVars x numFunctionals
    fVals : np.array of floats, numTimepoints x numFunctionals

    Returns
    -------
    xDotByModel : np.array of floats, numTimepoints x numVars

    """
    
    xDotByModel = np.zeros(x.shape)
    for i in range(x.shape[1]):
        xDotByModel[:, i] = np.sum(fVals * np.tile(coeffs[i, :], (fVals.shape[0], 1)), axis=1)
        
    return xDotByModel

# End of calculateDerivativesFromModel_fn
#----------------------------------------------------------------------

def estimateDerivatives_fn(xS, startInds, numDtStepsForDeriv, pointWtsForThisVar, dt):
    """
    For a vector, calculate the target derivative (as in dx/dt = rise/run) that we hope to match
    when we regress on the library functionals. The target derivative is calculated using the
    variables' time-series. Also return the weights for each element of the target rise, for use
    in the regression.
    NOTE: We require values in xS before the first startInd and beyond the last start ind

    Parameters
    ----------
    xS : np.array of floats, (column vector same length as timepoints eg tTrain), the
    pre-processed time-series of one variable.
    startInds : np.array (vector of ints). Indices of timepoints.
    numDtStepsForDeriv : scalar int
    pointWtsForThisVar :  np.array (vector of floats, same length as timepoints eg tTrain).
    dt : scalar float. The length of the simulation timestep

    Returns
    -------
    derivEstimate : np.array (vector of floats). size = startInds.shape
    weights : np.array (vector of floats). size = startInds.shape

    """
    timeStep = dt*numDtStepsForDeriv  # float (ie a measure of time, not an index of timepoints)

    # 4th order approx to derivative:
    m2Inds = startInds - 2*numDtStepsForDeriv  # indices
    m1Inds = startInds - 1*numDtStepsForDeriv
    p2Inds = startInds + 2*numDtStepsForDeriv
    p1Inds = startInds + 1*numDtStepsForDeriv
    derivEstimates = \
        (xS[m2Inds] - 8*xS[m1Inds] + 8*xS[p1Inds] - xS[p2Inds]) / (12*timeStep)
    # The weight for each deriv estimate combines the timepoint values used:
    weights = \
        pointWtsForThisVar[m2Inds] + 8 * pointWtsForThisVar[m1Inds] + \
            8 * pointWtsForThisVar[p1Inds] + pointWtsForThisVar[p2Inds]
    weights = weights / sum(weights)

    return derivEstimates, weights

# End of estimateDerivatives_fn
#--------------------------------------------------------

def defineXAndYForLinRegressionOnEvolution_fn(xT, functionalVals, startInds, numEvolveSteps,
                                              pointWtsForThisVar, dt):
    """
    Given start inds, create a target y to be fitted by a linear function of functionals by:
    1. define a set of subtargets subY which are individual estimates of derivatives over short
    hops, using a sequence of equally-spaced points.
    2. define an 'evolution' from the first to the last point by Euler stepping. It is not
    a real evolution because the input values of the points at each step are from the given time-
    series, not the prediction from the previous timepoint. We do this to maintain a linear
    relationship between the functionals and the target value:
    (x[t + n] - x[t])/dt = summation( beta_i * (f_i[t] + f_i[t+1] + ... + f_i[t+n-1]) ).
    So X = summations of the functionals over the n timepoints; and y = the difference between
    the two end timepoints, divided by the timestep.

    Parameters
    ----------
    xT : np.array of floats, numTimepoints x numVariables (the pre-processed time-series)
    functionalVals : np.array of floats, numTimepoints x num active functionals  for this
                     variable (the value of the active library functionals at each timepoint).
    startInds : np.array (vector of ints)
    numEvolveSteps : scalar int. How far to evolve to get regression target.
    pointWtsForThisVar : np.array (vector of floats, length = numTimepoints)
    dt : scalar float

    Returns
    -------
    X : np.array of floats, len(startInds) x num active functionals
    y : np.array (vector of floats, same size as 'startInds')
    weights : np.array (vector of floats, same size as 'startInds')
    """
    
    wtsPerPoint = np.zeros((numEvolveSteps, len(startInds)))
    evolveSteps = np.zeros((len(startInds), functionalVals.shape[1], numEvolveSteps))
    for i in range(numEvolveSteps):
        evolveSteps[:, :, i] = functionalVals[startInds + i, :]
        wtsPerPoint[i, :] = pointWtsForThisVar[startInds + i]

    X = np.sum(evolveSteps, axis=2)  # Sum over the evolve steps. The result is
    # len(startInds) x numFunctionals.
    y = (xT[startInds + 1] - xT[startInds]) / dt
    weights = np.median(wtsPerPoint, axis=0)
    weights = weights / np.sum(weights)  # so weights sum to 1
    return X, y, weights

# End of defineXAndYForLinRegressionOnEvolution_fn
#-----------------------------------------------------------

def combineSegmentCoeffs_fn(coeffsPerSeg, printDiagnosticsFlag, varName, fnalName,
                            snrThreshold, minNumSegmentResultsToUse, outputFilename):
    """
    Given a vector of coefficients (found by regressing on many segments, or subsets of
    timepoints) for some {variable, functional} pair, return (i) a median value, and (ii) a
    measure of reliability (stdDev/median). We do this by rejecting outliers until either a
    minimum number of coeffs are left, or the stdDev/median is within some bound.

    Parameters
    ----------
    coeffsPerSeg : np.array (vector of floats). length = number of segments that were used in
                   regressions.
    printDiagnosticsFlag : scalar boolean
    varName : str
    fnalName : str
    snrThreshold : scalar float
    minNumSegmentResultsToUse : scalar int (could be a float)
    outputFilename : str

    Returns
    -------
    coeff : scalar float
    stdOverMedian : scalar float

    """

    # Decide what coeff value to carry forward:
    # 1. If the SNR is low, just use mean or median.
    segsToUse = coeffsPerSeg != -100  # This should not be empty.
    this = coeffsPerSeg[segsToUse].copy()
    m = np.median(this)
    if m == 0:  # to catch an edge case
        m = np.mean(this)
    s = np.std(this)
    # 2. Start removing outliers until snr is low:
    while s/np.abs(m) > snrThreshold and len(this) > minNumSegmentResultsToUse:
        # Eliminate values and retry:
        dist = np.abs(this - m)
        trimmedThis = this[dist < max(dist)]
        if len(trimmedThis) == 0:  # Case: every remaining value is equidistant from the median, so
        # trimming would discard them all and leave a median of nan. Two values are always
        # equidistant from their median, so this is the norm when few segments are used. Keep the
        # untrimmed values and settle for their median.
            break
        this = trimmedThis
        m = np.median(this)
        if m == 0:
            m = np.mean(this)
        s = np.std(this)

    # Assign the final coeffs for this {variable, function} pair:
    coeff = np.median(this)
    stdOverMedian = np.abs(s / m)  # We'll cull functions based on high variability.

    # (Diagnostic) Print out vector of segment coeffs if not too many:
    if printDiagnosticsFlag:  # ie few enough fns in library that we can print the outcome:
        console = sys.stdout
        with open(outputFilename, 'a') as file:
            print(varName + "', " + fnalName + ' values: ' + \
                  str(np.round(coeffsPerSeg,2)) + ', median ' + \
                  str(np.round(np.median(coeffsPerSeg), 2)) + ', std ' + \
                  str(np.round(np.std(coeffsPerSeg), 2)) + '. Final median = ' + \
                  str(np.round(coeff, 2)) + ', final stdOverMedian = ' +  \
                  str(np.round(stdOverMedian, 2)), file=file)
            sys.stdout = console
            file.close()
        

    return coeff, stdOverMedian

# End of combineSegmentCoeffs_fn
#------------------------------------------------------------

def cullBasedOnStdOverMedian_fn(coeffArray, coeffsBySeg, stdOverMedianArray, functionsToUseArray,
                                startIndsToUse, segStartInds, segEndInds, pointWeights, 
                                functionalVals, p):
    """
    Cull functions whose fitted coeffs had high stdDev/Median, ie high variability over the
    various segments. This has two steps:
      1. Cull functionals based on high variability. Do not cull functionals with relatively high
         median (weighted) coefficients.
      2. Do a new regression, to calculate new coeffs for the remaining functionals.
    NOTE: This method gave bad results (though it is the culling criterion used in REF).
    
    Parameters
    ----------
    coeffArray : np.array of floats, numVariables x numFunctionals. The previous coefficients.
    coeffsBySeg : np.array of floats, numVariables x numFunctionals x numSegments. The coefficients 
                  from the new regressions on each segment, to be combined in this function.
    stdOverMedian : np.array of floats, numVariables x numFunctionals
    functionsToUseArray : np.array of booleans, numVariables x numFunctionals
    startIndsToUse : np.array of booleans, (numTimepoints - 2*margin) x numVars
    pointWeights : np.array of floats, numTimepoints x numVars
    functionalVals : np.array of floats, numTimepoints x numFunctionals
    p : dict of params, including: 
        xTrain : np.array of floats, numTimepoints x numVars
        minStdOverMedianThreshold : scalar float
        numFunctionsToTriggerCullBasedOnStdOverMean : scalar int
        maxNumToRemoveByStdPerCull : scalar int
        numDtStepsForDeriv : scalar int
        dt : scalar float (the timestep)
        regressOnFftAlpha : scalar float (the fraction of regression that is on FFTs)
        fftXDotTrainTarget : np.array (numFftPoints x numVars)
        fftLibFns : np.array (numFftPoints x numFunctionals)
        variableNames : list (np.array?) of str
        functionList : list (np.array?) of str
        margin : int
        weightTimepointsFlag : bool
        fftRegressionTarget : str
        snrThreshold : scalar float
        outputFilename : str

    Returns
    -------
    postRegressionCoeffs : np.array of floats, , numVariables x numFunctionals
    functionsToUseArray : np.array of booleans, numVariables x numFunctionals

    """

    # Loop through the variables, handling each one independently:
    for v in range(functionsToUseArray.shape[0]):
        # Check if any functions have high variance and also lower magnitude for this variable,
        # and also if there are few enough functions left to start this type of cull:
        # Calculate stdOverMedian threshold:
        tempStd = stdOverMedianArray[v, functionsToUseArray[v, :]]
        stdThreshold = np.median(tempStd) + 1 * np.std(tempStd)
        stdThreshold = max(stdThreshold, p['minStdOverMedianThreshold'])
        # Calculate coeff median threshold:
        tempMag = np.abs(coeffArray[v, functionsToUseArray[v, :]])
        magThreshold = np.median(tempMag)
        pointWtsForThisVar = pointWeights[:, v]

        # If there are no active functions, stdThreshold and medianThreshold = np.nan
        # Identify noisy functionals: high variability and low magnitude:
        noisyFnInds = np.where(np.logical_and(tempStd > stdThreshold, tempMag < magThreshold))[0]

        numActiveFns = np.sum(functionsToUseArray[v, :])
        if len(noisyFnInds) > 0 and numActiveFns <= \
            p['numFunctionsToTriggerCullBasedOnStdOverMean'] and numActiveFns > 2:
            # Cull only an allowable number of these. Argsort in descending order, then
            # cull the first several:
            inds = (np.argsort(stdOverMedianArray[v, noisyFnInds]))[-1: :-1]
            noisyFnInds = noisyFnInds[inds[0:p['maxNumToRemoveByStdPerCull']]]

            # (Diagnostic) Output to console:
            console = sys.stdout
            with open(p['outputFilename'], 'a') as file:
                print('Culled ' + str(np.array(p['functionList'])[noisyFnInds]) + \
                      ' from variable ' + p['variableNames'][v] + \
                      ' due to coeff stdOverMedian = ' + \
                      str(np.round(tempStd[noisyFnInds], 2)) + ' > ' + \
                      str(np.round(stdThreshold, 2)) + ' and coeff mag = ' + \
                      str(np.round(tempMag[noisyFnInds], 2)) + ' < ' + \
                      str(np.round(magThreshold, 2)) + '. Re-running linear regression.',file=file)
                sys.stdout = console
                file.close()

            # Update functionsToUseArray to cull noisy functions for this variable:
            functionsToUseArray[v, noisyFnInds] = False

            # Zero out coeffs for this variable. The relevant ones will be replaced using the new
            # regressions:
            coeffsBySeg[v, :, :] = -100
            coeffArray[v, :] = 0

            # Pick new segments and regress on each in turn. Use the same 'startIndsToUse' array.
            startIndsVarI = startIndsToUse[v,].copy()  # booleans
            # Convert to indices of timepoints:
            startIndsVarI  = np.where(startIndsVarI == True)[0] + p['margin']
            for seg in range(p['numRegressionSegments']):
                # Pick a subset of startIndsVarI  for this segment:
                if p['useRandomSegmentsFlag']:  # Case: Randomly choose points to regress on.
                    # Clamp to the number of available timepoints, since we sample without
                    # replacement below:
                    numPointsPerSegment = \
                        min(len(startIndsVarI),
                            int(len(startIndsVarI) / p['numRegressionSegments'] * \
                                (1 + 2 * p['overlapFraction'])))
                    startIndIndices = np.sort(np.random.choice(range(len(startIndsVarI)),
                                              numPointsPerSegment, replace=False))
                    startInds = startIndsVarI[startIndIndices]
                else:  # Case: Use sequential segments
                    startInds = startIndsVarI[np.logical_and(startIndsVarI >= segStartInds[seg],
                                                             startIndsVarI <= segEndInds[seg])]

                # Now do regression, if there are startInds in this segment:
                functionInds = np.where(functionsToUseArray[v, :] == True)[0]
                if len(startInds) > 0 and len(functionInds) > 0:  # Case: There are some startInds.
                    # Define the target rise based on the derivative approximation:
                    if p['regressionTarget'] == 'estimated derivatives':  # Currently always true
                        y, weights = \
                            estimateDerivatives_fn(p['xTrain'][:, v], startInds, 
                                                   p['numDtStepsForDeriv'], 
                                                   pointWtsForThisVar, p['dt'])

                        # Extract the relevant functions (columns):
                        X = functionalVals[startInds, :].copy()
                        X = X[:, functionInds]
                    if p['weightTimepointsFlag']:
                        sampleWeights = weights
                    else:
                        sampleWeights = np.ones(weights.shape) / len(weights)  # uniform, sum to 1

                    # Maybe also regress on fft(xDotTrain) as targets. There are many
                    # fewer regression points in the fft target: 50 vs. up to 5000 for the
                    # usual raw derivs, so use the sample weights to balance this out.
                    if p['regressOnFftAlpha'] > 0:
                        rawYAlpha = 1 - p['regressOnFftAlpha']
                        yFftOfXDot = p['fftXDotTrainTarget'][:, v]
                        y = np.hstack((y, yFftOfXDot ))  # Stack the targets
                        XForFft = p['fftLibFns'][:, functionInds].copy()
                        XForFft[np.where(np.isnan(XForFft))] = 0  # since fft of constant == nan
                        X = np.vstack((X, XForFft))
                        
                        numFftPoints = len(yFftOfXDot)
                        sampleWeights = \
                                np.hstack((rawYAlpha * sampleWeights, p['regressOnFftAlpha'] * \
                                           np.ones(numFftPoints) / numFftPoints))

                    # Finally ready to do the regressions on the segment 'seg':
                    # Special case: If we are regressing on complex FFT, we need to use lstsq:
                    if p['fftRegressionTarget'] == 'complex' and p['regressOnFftAlpha'] > 0:
                        w = sampleWeights.reshape(-1, 1)
                        betas = lstsq(np.sqrt(w) * X, np.sqrt(w) * y.reshape(-1, 1), rcond=-1)[0]
                        betas = np.real(betas)  # This may be very important and harmful.
                        coeffsBySeg[v, functionInds, seg] = betas.flatten()
                    else:
                        linReg = LinearRegression()
                        linReg = linReg.fit(X, y, sample_weight=sampleWeights)
                        coeffsBySeg[v, functionInds, seg] = linReg.coef_

            # Regressions have now been done on each segment for variable v.
            # For this variable, process the collections of coeffs on the different
            # segments to get a single set of coeffs.
            prs = coeffsBySeg.copy()
            # We are still in variable 'i'.
            for j in range(len(p['functionList'])):
                if functionsToUseArray[v, j]:  # Case: this function is in use for this
                # var, so we need to calculate a single coeff.
                    coeff, dummy = \
                        combineSegmentCoeffs_fn(prs[v, j, :], False, p['variableNames'][v],
                                                p['functionList'][j], p['snrThreshold'], 
                                                p['minNumSegmentResultsToUse'], 
                                                p['outputFilename'])  
                                                # Output to file, not console
                    coeffArray[v, j] = coeff
                else:  # Case: this functional has been culled for this variable.
                    pass
        # All functionals have new coeffs for this variable i, after 2nd fitting
        # without the functionals culled due to high stdDev / median.

    return coeffArray, functionsToUseArray

# End of cullBasedOnStdOverMedian_fn
#-------------------------------------------------------------------------------

def printWeightedCoeffModel_fn(coeffArray, functionsToUseArray, weightArray, variableNames,
                               functionList, outputFilename):
    """
    Print out a version of the sindy model that shows the weighted coefficients used when culling
    functionals. These weights tend to inflate the coefficients of functionals with relatively high
    values, and decrease the coefficients of functionals with relatively low values, because a
    functional with high values can tolerate a lower coefficient and still have the same impact as
    a functional with low values but a higher coefficient.

    Parameters
    ----------
    coeffArray : np.array, floats numVars x numFunctionals
    functionsToUseArray : np.array of booleans, numVariables x numFunctionals
    weightArray : np.array of floats, numVariables x numFunctionals
    variableNames : list of str
    functionList : list of str
    outputFilename : str

    Returns
    -------
    None.

    """ 
    weightedCoeffArray = weightArray * coeffArray  
    this = printModel_fn(weightedCoeffArray, variableNames, functionList)
    console = sys.stdout
    with open(outputFilename, 'a') as file:
        print('(Weighted coefficients:) \n' + this + '\n', file=file)
        file.close()
        sys.stdout = console 

# End of printWeightedCoefficientModel_fn
#--------------------------------------------------------------------------
 
def evolveModel_fn(coeffArray, recipes, initConds, timepoints):
    
    """
    Evolve odes using solve_ivp.    
    Because odeint can blow up and treat '0' coefficients as non-zero, we add some fuss to prevent
    this in case it can happen with solve_ivp. Method:
    Make a new vector of starting values, with culled variables values set to 0. Use this to
    simulate the system. Then tack on constant values for the culled variables. This approach 
    assumes that if a variable has been zeroed out, then all functionals involving that variable
    have been zeroed.
    
    Parameters:
    ----------
    coeffArray : np.array of floats, numVars x numFunctionals. Coeffs for the model to be evolved.
    recipes : list of lists, length = numFunctionals. The i'th list gives the indices of the 
             variables to be multiplied to make the i'th fnal.
    initConds : np.array of floats, vector 1 x numVars
    timepoints : np.array of floats, vector
    
    Returns:
    -------
    xEvolved : np.array of floats. numVars x len(timepoints)
    
    """
    
    # The function used by solve_ivp:
    def odeFn(t, s, coef, recipes): 
        """
        Return the values of s at the next step:

        Parameters
        ----------
        t : implicit variable
        s : np.array of floats, vector with len = numVars
        coef : np.array of floats, numVars x numFunctionals
        recipes : list of lists, each list may be an int (or np.int64), or a list of ints

        Returns
        -------
        df : np.array of floats, vector with len = numVars

        """
        # Note: 'coef' arrives already converted to a 2-D array (numVars x numFunctionals) by the
        # caller below, since re-converting it on every right-hand-side evaluation is pure overhead.
        fnals = np.zeros(len(recipes))
        for i in range(len(fnals)):
            r = recipes[i]
            if isinstance(r, (int, np.integer)):
                if r == -1:  # -1 means constant term
                    fnals[i] = 1 
                else:
                    fnals[i] = s[r]
            else:
                temp = 1
                for j in range(len(r)):
                    temp = temp * s[r[j]]
                fnals[i] = temp
        
        df = np.zeros(s.shape)
        for i in range(len(df)):
            df[i] = np.dot(fnals, coef[i, ])
         
        return df
     
    method =  'LSODA'  # 'RK45'. For stiff equations: ‘Radau’ or ‘BDF’
    startEnd = [timepoints[0], timepoints[-1]]
    argList = []
    argList.append(np.array(coeffArray).reshape(len(initConds), -1))  # converted once, not per step
    argList.append(recipes)
    
    # Fussing to prevent possible blow-ups of zeroed-out variables:
    # First remove any variables that have all zero coefficients:
    temp = np.abs(coeffArray) > 1e-5
    zeroedOutVars = np.where(np.sum(temp, axis=1) == 0)[0]  # The culled variables. 

    # Make a set of starting data with zeros in the culled variables: 
    initCondsCulledVarsZeroed = initConds.copy() 
    if len(zeroedOutVars) > 0:
        initCondsCulledVarsZeroed[zeroedOutVars] = 0

    xEvolved = np.zeros((len(timepoints), len(initConds)))  # Initialize
        
    sol = solve_ivp(odeFn, t_span=startEnd, y0=initCondsCulledVarsZeroed, method=method, 
                    t_eval=timepoints, args=argList)

    if sol.success:
        xEvolved = (sol.y).transpose()
    else:
        print('Error notice: solve_ivp failed.')
    
    # Fill in the constant vars:
    if len(zeroedOutVars) > 0:
        xEvolved[:, zeroedOutVars] = np.tile(initConds[zeroedOutVars], (len(timepoints), 1))
        
    return xEvolved  # could also return sol.t

# End of evolveModel_fn
#-------------------------------------------------------------------------

def generateVariableEnvelopes_fn(xData, numInWindow, dt, shrinkFactor=1):
    """
    For each point in a time-series, find the max and min values in a neighborhood for each
    variable.

    Parameters
    ----------
    xData : np.array of floats, numTimepoints x numVariables
    numInWindow : scalar int
    dt : scalar float
    shrinkFactor : scalar float >= 1

    Returns
    -------
    varLocalMax: np.array of floats, size xData.shape
    varLocalMin: np.array of floats, size xData.shape

    """
    # If required, shrink data towards local mean values:
    if shrinkFactor > 1:
        for i in range(xData.shape[1]):
            slopeX, stdX, meanX = calculateSlopeAndStd_fn(xData[:, i], dt, numInWindow)
            xData[:, i] = (xData[:, i] - meanX) / shrinkFactor + meanX

    half = int(np.floor(numInWindow / 2))

    varLocalMax = np.zeros(xData.shape)
    varLocalMin = np.zeros(xData.shape)

    for i in range(xData.shape[0]):  # loop over timepoints in this variable
        startInd = max(0, i - half)
        endInd = min(xData.shape[0], i + half)
        varLocalMax[i, :] = np.max(xData[startInd:endInd, :], axis=0)
        varLocalMin[i, :] = np.min(xData[startInd:endInd, :], axis=0)

    return varLocalMax, varLocalMin

# End of generateVariableEnvelopes_fn
#------------------------------------------------------------------------

def calculateFiguresOfMerit_fn(x, xTrue, fftPowerTrue, localMin, localMax, stdDevVector, 
                               meanVector, medianVector, fomTimepoints, maxPhaseShift):
    """
    Evolve a model and calculate various figures of merit.
    NOTE: We assume that xEvolved includes only timepoints in 'fomTimepointInds'. For 
    'inEnvelopeFoM' we take the max over small phase shifts of the whole time-series.

    Parameters
    ----------
    x : np.array of floats, numTimepoints x numVariables
    xTrue : np.array of floats, numTimepoints x numVariables
    fftPowerTrue :  np.array of floats, numFftPoints x numVariables 
    localMin : np.array of floats, numTimepoints x numVariables
    localMax : np.array of floats, numTimepoints x numVariables
    stdDevVector : np.array of floats, numTimepoints x numVariables
    meanVector : np.array of floats, numTimepoints x numVariables
    medianVector : np.array of floats, numTimepoints x numVariables
    fomTimepoints : np.array (list?) of ints
    maxPhaseShift : scalar int. Must be >= 1

    Returns
    -------
    fomDict: dict

    """
    if len(x.shape) == 1:  # Case: 1-dim with shape (val, ) rather than (val, 1)
        x = np.expand_dims(x, 1)

    phaseShifts = np.array(range(-maxPhaseShift, maxPhaseShift, 3))  # hop by 3s for speed
    if len(phaseShifts) == 0:
        phaseShifts = np.array([0])  # Ensure at least one phase shift value
    first = maxPhaseShift # Used to prevent overshooting.
    final = x.shape[0] - maxPhaseShift # Ditto.

    fomDict = dict()
    inBoundsFoM = np.zeros(x.shape[1])
    inEnvelopeFoM = np.zeros(x.shape[1])
    inEnvPhaseShiftVals = np.zeros(len(phaseShifts))
    globalMin = np.min(localMin, axis=0)
    globalMax = np.max(localMax, axis=0)
    for i in range(x.shape[1]):
        inBoundsFoM[i] = \
            np.sum(np.logical_and(x[:, i] > globalMin[i], 
                                  x[:, i] < globalMax[i])) / len(fomTimepoints)
        for j in range(len(phaseShifts)):
            pS = phaseShifts[j]
            inEnvPhaseShiftVals[j] = \
                np.sum(np.logical_and(x[first + pS:final + pS, i] > \
                                         localMin[first:final, i],
                                         x[first + pS:final + pS, i] < \
                                             localMax[first:final, i])) / len(fomTimepoints)
        inEnvelopeFoM[i] = max(inEnvPhaseShiftVals)

    # Statistics:
    temp = np.std(x, axis=0)
    stdDevFoM = (temp - stdDevVector) / stdDevVector
    stdDevFoM[stdDevFoM > 100] = 100 # To avoid uselessly big numbers
    temp = np.mean(x, axis=0)
    meanFoM = (temp - meanVector) / meanVector
    meanFoM[meanFoM > 100] = 100 # To avoid uselessly big numbers
    meanFoM[meanFoM < -100] = -100 # To avoid uselessly big numbers
    temp = np.median(x, axis=0)
    medianFoM = (temp - medianVector) / medianVector
    medianFoM[medianFoM > 100] = 100 # To avoid uselessly big numbers
    medianFoM[medianFoM < -100] = -100 # To avoid uselessly big numbers

    # Correlation of FFT:
    numFftPoints = fftPowerTrue.shape[0]
    fftPower = np.zeros([numFftPoints, x.shape[1]])
    fftCorrelationFoM = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        x1 = x[:, i].copy()
        x1 = x1 - np.mean(x1)
        xP = pow(np.real(np.fft.fft(x1)), 2) # power spectrum of fft
        temp = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])
        temp[np.isnan(temp)] = 0
        fftPower[:, i] = temp
        # Handle case that a var is identically 0:
        if not np.isnan(np.sum(fftPowerTrue[:, i])):
            fftCorrelationFoM[i] = np.dot(fftPower[:, i], fftPowerTrue[:, i]) / \
                np.dot(fftPowerTrue[:, i],  fftPowerTrue[:, i])  # Normalize
        else:
            fftCorrelationFoM[i] = 0

    # Correlation of histograms:
    numHistogramBins = 100
    histogramRange = np.array((np.min(xTrue, axis=0), np.max(xTrue, axis=0)))  # 2 x numVars
    theseHistograms = np.zeros([numHistogramBins, x.shape[1]])
    theseHistogramBins = theseHistograms.copy()
    histogramCorrelationFoM = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        histTrue = np.histogram(xTrue[:, i], bins=numHistogramBins)[0]
        temp = np.histogram(x[:, i], bins=numHistogramBins,
                            range=(histogramRange[0, i], histogramRange[1, i]))
        theseHistograms[:, i] = temp[0]
        theseHistogramBins[:, i] = temp[1][0:-1]
        histogramCorrelationFoM[i] = np.dot(histTrue, theseHistograms[:, i]) / \
            np.dot(histTrue, histTrue)

    # # Print to console:
    # print('inBoundsFoM = ' + str(np.round(inBoundsFoM,2)) + \
    #       ', inEnvelopeFoM = ' + str(np.round(inEnvelopeFoM, 2)) + \
    #       ', stdDevFoM = ' + str(np.round(stdDevFoM, 2)) + \
    #       ', meanFoM = ' + str(np.round(meanFoM, 2)) + \
    #       ', medianFoM = ' + str(np.round(medianFoM, 2)) + \
    #       ', fftCorrelationFoM = ' + str(np.round(fftCorrelationFoM, 2)) + \
    #       ', histogramCorrelationFoM = ' + str(np.round(histogramCorrelationFoM, 2)) + '\n')

    # Bundle FoMs into dict for return:
    fomDict['inBoundsFoM'] = inBoundsFoM
    fomDict['inEnvelopeFoM'] = inEnvelopeFoM
    fomDict['stdDevFoM'] = stdDevFoM
    fomDict['meanFoM'] = meanFoM
    fomDict['medianFoM'] = medianFoM
    fomDict['fftCorrelationFoM'] = fftCorrelationFoM
    fomDict['fftPower'] = fftPower
    fomDict['histogramCorrelationFoM'] = histogramCorrelationFoM
    fomDict['histograms'] = theseHistograms
    fomDict['histogramBins'] = theseHistogramBins

    return fomDict

# End of calculateFiguresOfMerit_fn
#--------------------------------------------------------------

def weightedLeastSquares_fn(X, y, sampleWeights, fitInterceptFlag):
    """
    Weighted least-squares fit of y on the columns of X.

    This reproduces, step for step, what
        LinearRegression(fit_intercept=fitInterceptFlag).fit(X, y, sample_weight=sampleWeights)
    does internally (centering via _preprocess_data, sqrt-weight rescaling via _rescale_data, then
    scipy's lstsq with sklearn's singular-value cutoff, then _set_intercept). It exists because the
    regressions here are small and numerous, so scikit-learn's per-call input validation costs
    several times more than the solve itself.

    NOTE: unlike scikit-learn, this does no input validation, so NaNs/infs in X or y propagate
    silently instead of raising.

    Parameters
    ----------
    X : np.array of floats, numPoints x numFeatures
    y : np.array of floats, vector of length numPoints
    sampleWeights : np.array of floats, vector of length numPoints (non-negative)
    fitInterceptFlag : bool

    Returns
    -------
    coeffs : np.array of floats, vector of length numFeatures
    intercept : float
    """
    if fitInterceptFlag:  # ie center X and y on their weighted means first
        xOffset = np.average(X, axis=0, weights=sampleWeights)
        yOffset = np.average(y, axis=0, weights=sampleWeights)
        xCentered = X - xOffset
        yCentered = y - yOffset
    else:
        xCentered = X
        yCentered = y

    # A weighted fit is an unweighted fit of the sqrt(weight)-scaled data:
    sqrtWeights = np.sqrt(sampleWeights)
    cond = max(X.shape) * np.finfo(X.dtype).eps  # cut-off ratio for small singular values
    coeffs = lstsqScipy(xCentered * sqrtWeights[:, np.newaxis], yCentered * sqrtWeights,
                        cond=cond)[0]

    if fitInterceptFlag:
        intercept = yOffset - np.dot(xOffset, coeffs)
    else:
        intercept = 0.0

    return coeffs, intercept

# End of weightedLeastSquares_fn
#--------------------------------------------------------------

def symmetricPartProjection_fn(A, gamma):
    """
    Project a square matrix onto {A : sym(A) <= gamma * I}, in Frobenius norm.

    sym(A) = (A + A') / 2 is eigendecomposed and its eigenvalues are clamped at 'gamma'. The skew
    part (A - A') / 2 is left untouched, since it does not affect x' * sym(A) * x and so cannot
    break boundedness. This is the projection used by trapping SINDy specialized to a purely linear
    model: with no quadratic terms the trapping condition on the shifted linear operator reduces to
    negative (semi-)definiteness of sym(A). See Kaptanoglu et al., Phys. Rev. Fluids 6:094401
    (2021), and the underlying theorem in Schlegel & Noack, JFM 765:325-352 (2015).

    Parameters
    ----------
    A : np.array of floats, numVars x numVars
    gamma : float, <= 0. The margin. 0 -> marginally bounded, so purely imaginary eigenvalues (ie
            undamped oscillators) survive. < 0 -> asymptotically stable.

    Returns
    -------
    np.array of floats, numVars x numVars, satisfying sym(result) <= gamma * I.
    """
    S = (A + A.transpose()) / 2
    K = (A - A.transpose()) / 2
    eigVals, eigVecs = np.linalg.eigh(S)

    return eigVecs @ np.diag(np.minimum(eigVals, gamma)) @ eigVecs.transpose() + K

# End of symmetricPartProjection_fn
#--------------------------------------------------------------

def spectralClamp_fn(A, gamma):
    """
    Push max Re eig(A) down to 'gamma' by moving ONLY the violating eigenvalues.

    Real Schur decomposition A = Q T Q', with T quasi-upper-triangular: 1x1 diagonal blocks are
    real eigenvalues, 2x2 blocks are complex conjugate pairs. Clamp the real part of each
    violating block -- a real block down to gamma, a complex pair by shifting both diagonal
    entries so the pair's real part (a + d) / 2 lands at gamma -- then reassemble.

    Contrast with 'symmetricPartProjection_fn', which is the analogous repair for the 'identity'
    norm: that one rewrites the whole symmetric part, this one leaves Q and the strictly
    triangular part of T untouched, so the eigenvectors and the entire non-normal structure of A
    survive and only the spectrum moves. That is what makes it a much closer target for the ridge
    ladder in 'stabilizeLinearModel_fn' when normType is 'spectral': the pull has less distance to
    cover, so a smaller lam suffices and less of the data fit is spent.

    NOTE: the result is dense in general -- Q T Q' fills in -- so this is useful as a ridge TARGET
    (the refit stays on the support) but not as a direct replacement for the coefficients. It is
    also not a projection onto a convex set: the set of matrices with max Re eig <= gamma is not
    convex, so 'nearest' is not claimed and alternating it with a support mask has no convergence
    guarantee. 'weightedSymmetricPartProjection_fn' is the convex surrogate used for that.

    Parameters
    ----------
    A : np.array of floats, numVars x numVars
    gamma : float, <= 0. See 'symmetricPartProjection_fn'.

    Returns
    -------
    np.array of floats, numVars x numVars, satisfying max Re eig(result) <= gamma.
    """
    T, Q = schurDecomposition(A, output='real')
    numVars = A.shape[0]

    i = 0
    while i < numVars:
        isTwoByTwoBlock = i + 1 < numVars and T[i + 1, i] != 0
        if isTwoByTwoBlock:
            realPart = (T[i, i] + T[i + 1, i + 1]) / 2
            if realPart > gamma:  # shift the pair, which keeps its imaginary part intact
                T[i, i] = T[i, i] - (realPart - gamma)
                T[i + 1, i + 1] = T[i + 1, i + 1] - (realPart - gamma)
            i += 2
        else:
            T[i, i] = min(T[i, i], gamma)
            i += 1

    return Q @ T @ Q.transpose()

# End of spectralClamp_fn
#--------------------------------------------------------------

def boundednessViolation_fn(A, normType):
    """
    How far the linear model xDot = A*x is from being bounded. <= gamma means the constraint holds.

    'identity'  -> max eig of sym(A). This is the trapping-SINDy / Schlegel-Noack condition: it
                   certifies boundedness via V(x) = ||x||^2, whose derivative is 2 x' sym(A) x. It
                   is sufficient but NOT necessary, and strongly conservative for non-normal A.
    'spectral'  -> max Re eig(A), the spectral abscissa. For a purely linear model this is the
                   exact condition: trajectories are bounded iff no eigenvalue has positive real
                   part (and those on the axis are semisimple, which a generic fitted A satisfies).
                   Equivalently, it is the trapping condition optimized over all weights P > 0,
                   since min over P > 0 of max eig sym(R A inv(R)), P = R'R, is the spectral
                   abscissa. So this is the same family of certificates, just not tied to P = I.

    Parameters
    ----------
    A : np.array of floats, numVars x numVars
    normType : str, 'identity' or 'spectral'

    Returns
    -------
    float. np.inf if A is not finite, so callers read it as 'constraint violated' rather than
    having to catch an exception.
    """
    if not np.all(np.isfinite(A)):
        return np.inf
    if normType == 'spectral':
        return float(np.linalg.eigvals(A).real.max())

    return float(np.linalg.eigvalsh((A + A.transpose()) / 2).max())

# End of boundednessViolation_fn
#--------------------------------------------------------------

def lyapunovMetricFactor_fn(A, gamma, innerMargin, maxConditionNumber=1e8, gapGrowthFactor=4.,
                            maxNumDampings=20):
    """
    Build a stable target from A and a weight P > 0 that makes that target strictly feasible.

    The trapping condition with a free weight is A'P + PA <= 2*gamma*P, ie d/dt V <= 2*gamma*V for
    V(x) = x'Px. 'boundednessViolation_fn' explains why fixing P = I is conservative: the infimum
    of max eig sym(A) over all such weights is the spectral abscissa, so P = I tests a condition
    that can be arbitrarily far from the truth for non-normal A. This picks a P adapted to a matrix
    we already know is stable, so that the feasible set is a wide convex neighbourhood of that
    matrix instead of the narrow set around zero that P = I gives.

    The target is 'spectralClamp_fn' of A, and P solves the shifted Lyapunov equation

        (target - gamma*I)' P + P (target - gamma*I) = -I,

    which has a unique P > 0 whenever max Re eig(target) < gamma. Rearranged, that says
    target'P + P*target = 2*gamma*P - I <= 2*gamma*P, so the target satisfies the condition at
    exactly 'gamma' with slack I -- it is strictly interior, which is the point.

    P is returned as the Cholesky factor R with P = R'R, plus its inverse, because
    A'P + PA <= 2*gamma*P is equivalent to sym(R A inv(R)) <= gamma*I: a congruence by the
    invertible R turns the weighted condition into the plain one, so every 'identity' routine in
    this file can be reused in the transformed coordinates.

    THE DAMPING LOOP. Solving at the tightest admissible clamp gives a P that is useless: the
    Lyapunov integral has time constant 1 / (gamma - innerMargin), so clamping at just below gamma
    sends cond(P) to 1e12 .. 1e17 on ordinary non-normal matrices. That is not only a numerical
    problem -- sqrt(cond(P)) is exactly the transient amplification the certificate permits, so an
    ill-conditioned P certifies almost nothing. The fix is to clamp the target further below gamma
    until cond(P) comes under 'maxConditionNumber', growing the gap geometrically from |gamma|. It
    is cheap in fit terms because the two quantities move at completely different rates: measured
    on random non-normal matrices ('_probe_lyapunov_conditioning.py'), growing the gap from 1e-3
    to 1e0 drops cond(P) by 5-6 orders of magnitude while the target moves from ~0.1-2 % of ||A||
    to only ~0.3-6 %.

    So 'maxConditionNumber' is the real knob of this whole approach, and it interpolates between
    the two settings of 'boundednessViolation_fn': at 1 it forces P = I (no transient growth
    allowed, maximally conservative), and as it grows the certificate approaches the spectral
    abscissa (any transient allowed, no useful bound on the trajectory). The default 1e8 permits
    an amplification of 1e4 and still leaves 8 digits in the transformed coordinates.

    Parameters
    ----------
    A : np.array of floats, numVars x numVars. The fitted matrix, not required to be stable.
    gamma : float, <= 0
    innerMargin : float, < gamma. The tightest clamp to try, ie the least damped target.
    maxConditionNumber : float, >= 1. See above.
    gapGrowthFactor : float, > 1. How fast the clamp margin walks below gamma.
    maxNumDampings : int

    Returns
    -------
    targetA : np.array of floats, numVars x numVars, with max Re eig <= innerMargin. The damped
              target when okFlag is True, the undamped clamp of A when it is False, so that the
              caller always has a usable ridge target.
    metricFactor : np.array of floats, numVars x numVars, upper triangular R with P = R'R, or None
    metricFactorInverse : np.array of floats, numVars x numVars, inv(R), or None
    numEigCalls : int
    okFlag : bool. False if no damping in range produced a well-conditioned positive definite P,
             in which case callers fall back to the P = I path.
    """
    numVars = A.shape[0]
    identity = np.eye(numVars)
    undampedTarget = spectralClamp_fn(A, innerMargin)
    numEigCalls = 1
    if not np.all(np.isfinite(A)):
        return undampedTarget, None, None, numEigCalls, False

    # Past this the target is essentially -gap*I and has nothing of A left in it, so stop rather
    # than keep damping:
    maxGap = 10 * (float(np.abs(np.linalg.eigvals(A)).max()) + abs(gamma))
    numEigCalls += 1

    gap = max(abs(gamma), 1e-12)
    for i in range(maxNumDampings):
        margin = min(innerMargin, gamma - gap)
        targetA = undampedTarget if i == 0 and margin == innerMargin else spectralClamp_fn(A,
                                                                                           margin)
        numEigCalls += 1

        try:
            P = solveContinuousLyapunov((targetA - gamma * identity).transpose(), -identity)
            P = (P + P.transpose()) / 2  # kill the asymmetry the solver leaves behind
        except (np.linalg.LinAlgError, ValueError):
            P = None
        numEigCalls += 1

        if P is not None and np.all(np.isfinite(P)):
            eigVals = np.linalg.eigvalsh(P)
            numEigCalls += 1
            if eigVals.min() > 0 and eigVals.max() / eigVals.min() <= maxConditionNumber:
                try:
                    metricFactor = np.linalg.cholesky(P).transpose()  # P = L L', so R = L'
                    metricFactorInverse = solveTriangular(metricFactor, identity, lower=False)
                except (np.linalg.LinAlgError, ValueError):
                    metricFactor, metricFactorInverse = None, None
                if metricFactor is not None and np.all(np.isfinite(metricFactorInverse)):
                    return targetA, metricFactor, metricFactorInverse, numEigCalls, True

        gap = gap * gapGrowthFactor
        if gap > maxGap:
            break

    return undampedTarget, None, None, numEigCalls, False

# End of lyapunovMetricFactor_fn
#--------------------------------------------------------------

def weightedSymmetricPartProjection_fn(A, gamma, metricFactor, metricFactorInverse):
    """
    Project A onto {A : A'P + PA <= 2*gamma*P}, in the norm ||R (.) inv(R)||_F, P = R'R.

    Change coordinates with B = R A inv(R), clamp sym(B) at gamma exactly as
    'symmetricPartProjection_fn' does, and change back. The congruence identity in
    'lyapunovMetricFactor_fn' makes this the exact projection onto the weighted set -- in the
    weighted norm, which is the one that matches the certificate. metricFactor = I recovers
    'symmetricPartProjection_fn'.

    Parameters
    ----------
    A : np.array of floats, numVars x numVars
    gamma : float, <= 0
    metricFactor : np.array of floats, numVars x numVars, upper triangular R with P = R'R
    metricFactorInverse : np.array of floats, numVars x numVars, inv(R)

    Returns
    -------
    np.array of floats, numVars x numVars, satisfying result'P + P*result <= 2*gamma*P, and hence
    max Re eig(result) <= gamma.
    """
    B = symmetricPartProjection_fn(metricFactor @ A @ metricFactorInverse, gamma)

    return metricFactorInverse @ B @ metricFactor

# End of weightedSymmetricPartProjection_fn
#--------------------------------------------------------------

def weightedBoundednessViolation_fn(A, metricFactor, metricFactorInverse):
    """
    How far xDot = A*x is from A'P + PA <= 2*gamma*P, ie the smallest gamma that holds, P = R'R.

    This is max eig sym(R A inv(R)), the numerical abscissa in the P metric. It sits between the
    two settings of 'boundednessViolation_fn': >= max Re eig(A) always, and equal to
    max eig sym(A) when P = I. Being <= gamma is sufficient for max Re eig(A) <= gamma, so a model
    feasible here is feasible under normType 'spectral'.

    Parameters
    ----------
    A : np.array of floats, numVars x numVars
    metricFactor : np.array of floats, numVars x numVars, upper triangular R with P = R'R
    metricFactorInverse : np.array of floats, numVars x numVars, inv(R)

    Returns
    -------
    float. np.inf if the transformed matrix is not finite, matching 'boundednessViolation_fn'.
    """
    if not np.all(np.isfinite(A)):
        return np.inf

    return boundednessViolation_fn(metricFactor @ A @ metricFactorInverse, 'identity')

# End of weightedBoundednessViolation_fn
#--------------------------------------------------------------

def maskedRidgeTowardTarget_fn(coeffArray, targetCoeffArray, functionsToUseArray, designMatrices,
                               lam):
    """
    Refit each variable's coefficients on its existing support, with an L2 pull toward a target.

    For each variable v, solve
        min_b  sum_i w_i * (y_i - X_i * b)^2  +  lam * ||b - target_v||^2
    over the active functionals of v only. Restricting to the active support is what stops the
    stabilizing projection from filling in functionals that the cull loop has already removed.

    The sample weights are renormalized to sum to the number of timepoints, so that 'lam' is
    calibrated against a Gram matrix of the usual unweighted magnitude and the same ladder of
    lam values means the same thing regardless of how many timepoints a variable happens to have.

    Parameters
    ----------
    coeffArray : np.array of floats, numVars x numFunctionals. Only its inactive entries are used
                 (they stay zero); active entries are overwritten.
    targetCoeffArray : np.array of floats, numVars x numFunctionals. The pull target.
    functionsToUseArray : np.array of bools, numVars x numFunctionals
    designMatrices : dict, keyed by variable index. Each value is a dict with keys 'X' (numPoints x
                     numActiveFunctionals, columns ordered as np.where(functionsToUseArray[v])[0]),
                     'y' (length numPoints), 'sampleWeights' (length numPoints), and 'functionInds'.
    lam : float, > 0

    Returns
    -------
    np.array of floats, numVars x numFunctionals
    """
    newCoeffArray = np.zeros(coeffArray.shape)
    for v in range(coeffArray.shape[0]):
        if v not in designMatrices:  # Case: this variable had no usable functionals when the
        # design matrices were cached, so leave its row at zero.
            continue
        d = designMatrices[v]
        functionInds = d['functionInds']
        # The cull steps can deactivate functionals after the design matrix was cached, so keep
        # only the columns that are still active:
        keep = functionsToUseArray[v, functionInds]
        if not np.any(keep):
            continue
        X = d['X'][:, keep]
        cols = functionInds[keep]

        w = d['sampleWeights']
        w = w * (len(w) / np.sum(w))  # renormalize to sum to numPoints, see docstring
        Xw = X * w[:, np.newaxis]
        H = Xw.transpose() @ X + lam * np.eye(X.shape[1])
        b = Xw.transpose() @ d['y'] + lam * targetCoeffArray[v, cols]
        newCoeffArray[v, cols] = np.linalg.solve(H, b)

    return newCoeffArray

# End of maskedRidgeTowardTarget_fn
#--------------------------------------------------------------

def projectOntoBoundedAndSupport_fn(A, supportMask, gamma, maxNumIters=300, tol=1e-12,
                                    metricFactor=None, metricFactorInverse=None):
    """
    Find a matrix that both satisfies the boundedness condition and has zeros off 'supportMask'.

    Alternating projections between the two sets: clamp the eigenvalues of the symmetric part (see
    'symmetricPartProjection_fn'), then zero the entries outside the support, and repeat. Both sets
    are convex, so this converges to a point in their intersection whenever one exists.

    NOTE: a single projection followed by masking does NOT work, which is the reason this iteration
    exists. Zeroing entries of an already-projected matrix destroys the eigenvalue clamp, so the
    masked projection is generally still unbounded; empirically its largest symmetric eigenvalue
    lands well above gamma. Alternating restores both properties together.

    NOTE: the intersection is non-empty for gamma = 0 (A = 0 is in it), but for gamma < 0 it
    requires the diagonal of A to be inside the support, since sym(A) <= gamma*I < 0 forces
    A[i, i] < 0. Callers must ensure this; the iteration simply fails to converge otherwise.

    Passing a metric factor switches the condition from sym(A) <= gamma*I to the weighted
    A'P + PA <= 2*gamma*P, P = R'R, using 'weightedSymmetricPartProjection_fn' in place of the
    plain one. That set is still convex, so the argument above survives, with one caveat: the two
    projections are then orthogonal in DIFFERENT inner products (the weighted one in
    ||R (.) inv(R)||_F, masking in Frobenius), so von Neumann's theorem no longer applies and
    convergence is empirical. The feasibility test below is what decides, and the caller has a
    fallback for 'convergedFlag' False, so a failure here degrades rather than corrupts.

    Parameters
    ----------
    A : np.array of floats, numVars x numVars
    supportMask : np.array of bools, numVars x numVars
    gamma : float, <= 0
    maxNumIters : int
    tol : float. Slack allowed when testing the result, to absorb round-off in the
          eigendecomposition.
    metricFactor : np.array of floats, numVars x numVars, or None for P = I. Upper triangular R
                   with P = R'R, as returned by 'lyapunovMetricFactor_fn'.
    metricFactorInverse : np.array of floats, numVars x numVars, or None. inv(R).

    Returns
    -------
    B : np.array of floats, numVars x numVars, with zeros off 'supportMask'
    convergedFlag : bool
    numEigCalls : int
    """
    weightedFlag = metricFactor is not None and metricFactorInverse is not None

    # The clamp lands the iterate exactly ON the boundary, so round-off decides the feasibility
    # test. In the plain metric that is 1e-16 and 'tol' absorbs it; in the weighted one it is
    # amplified by cond(P), which 'lyapunovMetricFactor_fn' allows up to 1e8, and the test would
    # fail by ~1e-9 every time. So aim the projection strictly inside by that much, the same
    # innerMargin idea 'trappingLinearModel_fn' uses:
    projectionMargin = gamma
    if weightedFlag:
        metricConditionNumber = float(np.linalg.cond(metricFactor)) ** 2  # = cond(P)
        projectionMargin = gamma - 8 * np.finfo(float).eps * metricConditionNumber * \
            max(1., abs(gamma))

    def violation_fn(M):
        if weightedFlag:
            return weightedBoundednessViolation_fn(M, metricFactor, metricFactorInverse)
        return boundednessViolation_fn(M, 'identity')

    def projection_fn(M):
        if weightedFlag:
            return weightedSymmetricPartProjection_fn(M, projectionMargin, metricFactor,
                                                      metricFactorInverse)
        return symmetricPartProjection_fn(M, projectionMargin)

    B = A * supportMask
    # The iteration converges only if the intersection is non-empty. It is for gamma = 0 (B = 0 is
    # in it), but for gamma < 0 it needs enough freedom in the support, so bail out if the iterate
    # runs away rather than looping into an overflow:
    maxAllowedMagnitude = 1e6 * max(np.abs(B).max(), 1e-30)
    numEigCalls = 0
    for i in range(maxNumIters):
        numEigCalls += 1
        if not np.all(np.isfinite(B)) or np.abs(B).max() > maxAllowedMagnitude:
            return B, False, numEigCalls
        if violation_fn(B) <= gamma + tol:
            return B, True, numEigCalls
        B = projection_fn(B) * supportMask
        numEigCalls += 1  # the projection does an eigendecomposition too

    numEigCalls += 1
    convergedFlag = violation_fn(B) <= gamma + tol

    return B, convergedFlag, numEigCalls

# End of projectOntoBoundedAndSupport_fn
#--------------------------------------------------------------

def stabilizeLinearModel_fn(coeffArray, functionsToUseArray, designMatrices, numVars,
                            boundednessMargin, strictBoundednessMargin, ridgeLadder, variableNames,
                            outputFilename, normType='spectral', repairMetric='symmetric'):
    """
    Force a purely linear model to produce bounded trajectories, trapping-SINDy style.

    The model is xDot = c + A*x, where c = coeffArray[:, 0] (the constant functional) and
    A = coeffArray[:, 1:1+numVars] (the degree-1 monomials, which
    'generatePolynomialLibrary_fn' always emits in variable order directly after the constant).
    The constraint imposed is

        A'P + PA <= 2 * gamma * P,   gamma <= 0,  for some P > 0,

    which gives d/dt V(x) <= 0 for V(x) = x' * P * x, so V is non-increasing and every trajectory is
    trapped inside the level set of V through its initial condition. This is the trapping theorem's
    condition on the linear operator: with no quadratic terms the trapping-region center 'm' drops
    out and nothing else is left to constrain.

    'normType' says how P is handled; see 'boundednessViolation_fn'.
      'identity' -> P is fixed to I, ie literally the trapping-SINDy / Schlegel-Noack condition
                    sym(A) <= gamma*I.
      'spectral' -> P is left free, which reduces the test to the spectral abscissa. Default,
                    because P = I is strongly conservative for non-normal A, and that is not a
                    corner case here: 'core/matrix.py::generate_matrix' plants purely imaginary
                    eigenvalues but has max eig sym(A) of order +1, so under 'identity' the
                    constraint excludes the true model. Measured on those systems, 'identity' takes
                    the derivative-fit R^2 from ~0.34 to ~-31, while 'spectral' keeps it at ~0.30.

    NOTE on gamma: at gamma = 0 undamped oscillators (purely imaginary eigenvalues) remain
    admissible, which is what we want from a boundedness constraint. gamma < 0 is needed only when
    the constant functional is active, since then d/dt V / 2 = x'Pc + x'*sym(PA)*x can grow along
    the kernel; with gamma < 0 the trajectory is trapped in a ball.

    Four steps, in escalating order of damage to the data fit:
      1. If the constraint already holds, do nothing (one eigendecomposition). This is the common
         case once the model has settled, so it stays cheap.
      2. Refit on the existing support with an L2 pull toward a stabilized target, walking
         'ridgeLadder' and taking the first rung that satisfies the constraint. Refitting on the
         support is what stops the repair from filling in functionals the cull loop removed.
      3. If no rung suffices, project onto the feasible set intersected with the support, via
         'projectOntoBoundedAndSupport_fn'. Support-preserving. Needs the self-terms A[v, v] to be
         in the library; any that were culled are restored, and that is logged, since it partly
         undoes the cull loop.
      4. Diagonal shift, which satisfies the constraint by construction, if even that failed.

    'repairMetric' says which metric steps 2 and 3 REPAIR in, as opposed to 'normType', which says
    which condition is TESTED. The two were originally the same, and that is the source of most of
    the damage this function does at normType='spectral': it accepts the weak condition but aims
    every repair at the strong one, so a model that only needed its spectrum nudged gets its whole
    symmetric part rewritten.
      'symmetric' -> the original behaviour, and the default so that existing runs are unchanged.
                     Step 2 aims at 'symmetricPartProjection_fn' of A, step 3 projects onto
                     {sym(A) <= gamma}. Required when normType is 'identity', since that is the
                     condition being tested; the other two settings cannot satisfy it.
      'schur'     -> step 2 aims at 'spectralClamp_fn' of A instead, which moves only the violating
                     eigenvalues and leaves the eigenvectors and the non-normal structure alone.
                     The target is far closer to A, so an earlier, gentler rung of 'ridgeLadder'
                     becomes feasible. Step 3 unchanged.
      'lyapunov'  -> as 'schur', plus step 3 projects onto the weighted set A'P + PA <= 2*gamma*P
                     with P built from the clamped target by 'lyapunovMetricFactor_fn'. The
                     clamped target is then strictly interior to that set, so the projection has a
                     wide convex region to land in rather than the narrow one around P = I. Falls
                     back to the P = I projection if the Lyapunov solve does not return a positive
                     definite P.
    Both new settings imply max Re eig(A) <= gamma but NOT sym(A) <= gamma*I, so they are usable
    only at normType='spectral'; asking for them at 'identity' logs an error and uses 'symmetric'.

    MEASURED, AND THE REASON 'symmetric' IS STILL THE DEFAULT. The two new settings do what they
    were built to do and it does not pay off on trajectories:
      - planted linear systems, sparse support ('benchmark_trapping_vs_ladder.py'). 'schur' wins
        the ladder in 5 of 15 cases and falls back to the symmetric target in the rest, so median
        derivative R2 improves from -0.97 to -0.26. Median trajRmse is a wash (0.587 -> 0.591) and
        the WORST trajRmse degrades 2.3x (0.671 -> 1.574), every degraded case being one where the
        clamped target won.
      - the real loyalty series, dense support ('_probe_repair_metric_loyalty.py', 70 % train,
        hamming 19, free run scored on the held-out tail). v3/87: rmseTail 0.039 -> 72.5 and peak
        |x| 0.99 -> 579x the data's own range. continuous/26: derivative R2 improves 3000x
        (-2.5e5 -> -78.6) and rmseTail still degrades 10x, 0.230 -> 2.44, peak 0.40 -> 2.86.
      - 'lyapunov' never once differed from 'schur'. Its weighted projection failed to converge in
        0 of 15 synthetic cases (mismatched inner products, see 'projectOntoBoundedAndSupport_fn')
        and never even runs on the loyalty data, where the ladder succeeds first.
    The mechanism is the one the certificate advertises. P = I certifies ||x(t)|| <= ||x0||, which
    is why every 'symmetric' row above keeps peak |x| at or below the data's own range. The
    spectral condition permits transient growth of cond(eigenvectors) before the decay starts, and
    'spectralClamp_fn' preserves exactly the non-normal structure that produces it -- so the
    repaired model is bounded as promised and still leaves the data range by a factor of 579. The
    conservatism of P = I is doing load-bearing work here, not wasting fit.
    Use 'schur' when the derivative fit is the deliverable, which is the same conclusion the
    benchmark reaches for 'trapping' over the ladder. For trajectories, keep 'symmetric'.

    Parameters
    ----------
    coeffArray : np.array of floats, numVars x numFunctionals
    functionsToUseArray : np.array of bools, numVars x numFunctionals. Not modified; steps 3 and 4
                          return an updated copy.
    designMatrices : dict, see 'maskedRidgeTowardTarget_fn'. Pass {} to skip step 2.
    numVars : int
    boundednessMargin : float, <= 0. The gamma to use if the model has no constant term.
    strictBoundednessMargin : float, < 0. Used in place of 'boundednessMargin' when the latter is 0
                              but the constant functional is active. See the NOTE above. Also used
                              as the margin of the ridge target, so that a finite pull toward it can
                              land strictly inside the feasible set.
    ridgeLadder : list-like of floats, increasing
    variableNames : list-like of str
    outputFilename : str
    normType : str, 'spectral' or 'identity'. See above.
    repairMetric : str, 'symmetric', 'schur' or 'lyapunov'. See above. Default 'symmetric', which
                   is the pre-existing behaviour.

    Returns
    -------
    coeffArray : np.array of floats, numVars x numFunctionals
    functionsToUseArray : np.array of bools, numVars x numFunctionals
    actionStr : str, which of the four steps was used. At repairMetric other than 'symmetric' the
                metric is appended, eg 'ridge(lam=1.0,schur)', so a log says which repair ran.
    numEigCalls : int
    """
    linearInds = np.arange(1, 1 + numVars)  # columns of coeffArray holding A

    # Pick gamma. An active constant functional means an affine term, which can grow along the
    # kernel of sym(A), so marginal boundedness is not enough and we tighten to a strict margin:
    gamma = boundednessMargin
    if gamma == 0 and np.any(functionsToUseArray[:, 0]):
        gamma = strictBoundednessMargin

    # Guard: this only makes sense if the active library is {constant, degree-1 monomials}:
    if functionsToUseArray.shape[1] > 1 + numVars and \
        np.any(functionsToUseArray[:, 1 + numVars:]):
        print('Error: stabilizeLinearModel_fn was called with active functionals of degree >= 2. ' +
              'The boundedness constraint is implemented for linear models only. Skipping it.')
        return coeffArray, functionsToUseArray, 'skipped-nonlinear-library', 0

    if repairMetric not in ('symmetric', 'schur', 'lyapunov'):
        print('Error: repairMetric must be "symmetric", "schur" or "lyapunov", but is ' +
              str(repairMetric) + '. Using "symmetric".')
        repairMetric = 'symmetric'
    if normType == 'identity' and repairMetric != 'symmetric':
        # Neither of the two spectral repairs produces sym(A) <= gamma*I, so at this normType they
        # would simply never be accepted and every call would fall through to the fallbacks:
        print('Error: repairMetric "' + repairMetric + '" cannot satisfy normType "identity", ' +
              'which tests sym(A) <= gamma*I. Using "symmetric".')
        repairMetric = 'symmetric'
    metricSuffix = '' if repairMetric == 'symmetric' else ',' + repairMetric

    A = coeffArray[:, linearInds]
    violation = boundednessViolation_fn(A, normType)
    numEigCalls = 1
    if violation <= gamma:  # Step 1: nothing to do.
        return coeffArray, functionsToUseArray, 'already-bounded', numEigCalls

    # Step 2: refit on the current support with an L2 pull toward a stabilized target. The target
    # aims strictly inside the feasible set rather than at gamma, which is what lets a finite pull
    # land inside it too:
    targetMargin = min(gamma, strictBoundednessMargin)
    metricFactor, metricFactorInverse = None, None
    symmetricTarget = (symmetricPartProjection_fn(A, targetMargin), '', tuple(ridgeLadder))
    if repairMetric == 'symmetric':
        targetCandidates = [symmetricTarget]
    else:
        # A margin strictly below 'targetMargin', so that the clamped target has room to be an
        # interior point of the set tested at gamma -- which is also what the Lyapunov shift needs
        # in order to have a Hurwitz argument:
        innerMargin = targetMargin - max(abs(targetMargin), 1e-6)
        # Both metrics use the DAMPED clamp, not the bare one at 'innerMargin'. A bare clamp is
        # only ~1 % of ||A|| away from A, and 'maskedRidgeTowardTarget_fn' then masks the dense
        # Q T Q' back onto the support, which destroys a clamp that shallow -- measured, every
        # rung of the ladder stays infeasible and the repair degenerates into step 3. The damping
        # loop is what makes the target survive masking, so it earns its keep twice.
        targetA, metricFactor, metricFactorInverse, ne, okFlag = lyapunovMetricFactor_fn(
            A, gamma, innerMargin)
        numEigCalls += ne
        if repairMetric == 'schur':
            metricFactor, metricFactorInverse = None, None  # target only, P = I in step 3
        elif not okFlag:
            # Possible when A is so non-normal that no admissible damping gives a usable P. The
            # P = I projection is still a valid repair, just the conservative one, so degrade to
            # it rather than give up:
            with open(outputFilename, 'a') as file:
                print('Boundedness constraint: no well-conditioned Lyapunov weight was found, ' +
                      'falling back to the P = I projection.', file=file)
            metricSuffix = ',schur'
        # The clamped target needs a HARDER pull than 'ridgeLadder' reaches, and for the same
        # reason it is a better target: it sits ~1 % from A in Frobenius norm where the
        # symmetric-part projection sits ~50 % away, so an equal lam moves A ~50x less. Measured
        # on the loyalty library (dense support, 87 variables): the symmetric target is feasible
        # already at lam = 1, the clamped one only from lam ~ 1e6, and the path between is not
        # even monotone -- at lam = 1 the refit is MORE unstable than the free fit. So extend the
        # ladder for this candidate instead of letting it lose by default.
        extendedLadder = tuple(ridgeLadder) + tuple(ridgeLadder[-1] * factor
                                                    for factor in (1e2, 1e4))
        # Two targets, not one. Even damped, the clamp stays infeasible after a SPARSE support
        # mask in about two thirds of the planted cases, and no lam rescues those, since as lam
        # grows the refit converges to the masked target itself. But where it does land it is
        # worth a lot (see the benchmark), so try it first and keep the blunt target as the
        # fallback rather than letting a miss fall through to the projection. The ladder is a
        # handful of solves against the projection's 100-400 eigendecompositions, so the second
        # pass is cheap insurance.
        targetCandidates = [(targetA, metricSuffix, extendedLadder), symmetricTarget]

    for candidateTarget, targetSuffix, candidateLadder in targetCandidates:
        targetCoeffArray = coeffArray.copy()
        targetCoeffArray[:, linearInds] = candidateTarget
        for lam in candidateLadder:
            candidateCoeffArray = maskedRidgeTowardTarget_fn(
                coeffArray, targetCoeffArray, functionsToUseArray, designMatrices, lam)
            numEigCalls += 1
            if boundednessViolation_fn(candidateCoeffArray[:, linearInds], normType) <= gamma:
                return candidateCoeffArray, functionsToUseArray, \
                    'ridge(lam=' + str(lam) + targetSuffix + ')', numEigCalls

    # Step 3: project onto the feasible set intersected with the support. At repairMetric
    # 'symmetric' / 'schur' that set is {sym(A) <= gamma}, which implies max Re eig(A) <= gamma and
    # so satisfies either normType; at 'lyapunov' it is the weighted {A'P + PA <= 2*gamma*P}, which
    # implies max Re eig(A) <= gamma only -- hence the normType guard above.
    # Try the support as it stands first. Only if that fails do we restore culled self-terms, since
    # every restoration is a functional forced back into a model the cull loop had rejected:
    functionsToUseArray = functionsToUseArray.copy()
    # The weighted projection alternates two projections that are orthogonal in DIFFERENT inner
    # products (see 'projectOntoBoundedAndSupport_fn'), so unlike the P = I one it can fail to
    # converge. Retry at P = I before resorting to the diagonal shift: that is the pre-existing
    # repair, it converges here, and it is a strictly better floor than step 4.
    metricAttempts = [(metricFactor, metricFactorInverse)]
    if metricFactor is not None:
        metricAttempts.append((None, None))

    boundedA, convergedFlag, restoredFlag = A, False, False
    for attemptFactor, attemptFactorInverse in metricAttempts:
        boundedA, convergedFlag, ne = projectOntoBoundedAndSupport_fn(
            A, functionsToUseArray[:, linearInds], gamma, metricFactor=attemptFactor,
            metricFactorInverse=attemptFactorInverse)
        numEigCalls += ne

        if not convergedFlag and not restoredFlag:
            # Driving sym(A) strictly negative needs a negative diagonal, so the self-terms have
            # to be available. Restore the missing ones and retry, and say so: the boundedness
            # constraint overriding the cull loop is a real interaction, not a detail.
            restoredVars = []
            for v in range(numVars):
                if not functionsToUseArray[v, 1 + v]:
                    functionsToUseArray[v, 1 + v] = True
                    restoredVars.append(variableNames[v])
            restoredFlag = True
            if len(restoredVars) > 0:
                with open(outputFilename, 'a') as file:
                    print('Boundedness constraint: restored culled self-term(s) for ' +
                          ', '.join(restoredVars) +
                          ', since a bounded model needs a negative diagonal.', file=file)
                boundedA, convergedFlag, ne = projectOntoBoundedAndSupport_fn(
                    A, functionsToUseArray[:, linearInds], gamma, metricFactor=attemptFactor,
                    metricFactorInverse=attemptFactorInverse)
                numEigCalls += ne

        if convergedFlag:
            if attemptFactor is None and metricFactor is not None:
                metricSuffix = ',schur'  # log the metric that actually produced the result
            break

    if convergedFlag:
        projectedCoeffArray = coeffArray.copy()
        projectedCoeffArray[:, linearInds] = boundedA
        # 'metricSuffix' rather than 'repairMetric', so that a Lyapunov weight that failed and
        # degraded to the P = I projection is logged as what actually ran:
        return projectedCoeffArray, functionsToUseArray, \
            'projection' + ('(' + metricSuffix[1:] + ')' if metricSuffix else ''), numEigCalls

    # Step 4: diagonal shift. sym(A - s*I) = sym(A) - s*I and the eigenvalues shift with it, so
    # s = violation - gamma + epsilon satisfies either normType by construction:
    shift = violation - gamma + 0.01
    coeffArray = coeffArray.copy()
    for v in range(numVars):
        coeffArray[v, 1 + v] = coeffArray[v, 1 + v] - shift
    with open(outputFilename, 'a') as file:
        print('Boundedness constraint: ridge ladder and projection both failed, applied a ' +
              'diagonal shift of ' + str(np.round(shift, 4)) + ' instead.', file=file)

    return coeffArray, functionsToUseArray, 'diagonalShift', numEigCalls

# End of stabilizeLinearModel_fn
#--------------------------------------------------------------

def trappingPrecompute_fn(coeffArray, functionsToUseArray, designMatrices, numVars,
                          buildSymMapFlag=True):
    """
    Build the pieces of the trapping coefficient update that depend on neither nu nor auxA.

    Everything here costs O(numFree^2) or more and is identical across nu values and across
    iterations of the alternation, so it is built once by 'trappingLinearModel_fn' and handed to
    'trappingAlternate_fn'.

    'penaltyLinearModel_fn' reuses the data pieces but needs neither 'symMap' nor
    'penaltyHessian' -- it differentiates the penalty with respect to A directly, see
    'boundednessPenaltyAndGradient_fn' -- and 'penaltyHessian' alone is a numFree x numFree
    product of numVars^2 x numFree factors, so it passes buildSymMapFlag=False to skip both.

    Parameters
    ----------
    coeffArray : np.array of floats, numVars x numFunctionals
    functionsToUseArray : np.array of bools, numVars x numFunctionals
    designMatrices : dict, see 'maskedRidgeTowardTarget_fn'
    numVars : int
    buildSymMapFlag : bool. False -> 'symMap' and 'penaltyHessian' are returned as None.

    Returns
    -------
    dict with keys:
      'freeIndices'    : list of (variable, functional) pairs, the active entries, in solve order
      'symMap'         : np.array, (numVars * numVars) x numFree. Linear map from the free entries
                         to vec(sym(A)). The constant column of the library maps to zero, so the
                         affine term is untouched by the penalty.
      'dataHessian'    : np.array, numFree x numFree. 2 * X'WX, block diagonal over variables.
      'dataRhs'        : np.array, numFree. 2 * X'Wy.
      'penaltyHessian' : np.array, numFree x numFree. symMap' symMap.
    """
    freeIndices = [(v, f) for v in range(coeffArray.shape[0])
                   for f in range(coeffArray.shape[1]) if functionsToUseArray[v, f]]
    positionOf = {vf: k for k, vf in enumerate(freeIndices)}
    numFree = len(freeIndices)

    # vec(sym(A))[i * numVars + j] = (A[i, j] + A[j, i]) / 2, and A[i, j] is the coefficient of
    # variable i on functional 1 + j:
    symMap = None
    if buildSymMapFlag:
        symMap = np.zeros((numVars * numVars, numFree))
        for i in range(numVars):
            for j in range(numVars):
                rowOfMap = i * numVars + j
                if (i, 1 + j) in positionOf:
                    symMap[rowOfMap, positionOf[(i, 1 + j)]] += 0.5
                if (j, 1 + i) in positionOf:
                    symMap[rowOfMap, positionOf[(j, 1 + i)]] += 0.5

    dataHessian = np.zeros((numFree, numFree))
    dataRhs = np.zeros(numFree)
    for v in range(coeffArray.shape[0]):
        if v not in designMatrices:
            continue
        d = designMatrices[v]
        functionInds = d['functionInds']
        keep = functionsToUseArray[v, functionInds]
        if not np.any(keep):
            continue
        X = d['X'][:, keep]
        positions = [positionOf[(v, f)] for f in functionInds[keep]]
        w = d['sampleWeights']
        w = w * (len(w) / np.sum(w))  # same renormalization as maskedRidgeTowardTarget_fn
        Xw = X * w[:, np.newaxis]
        dataHessian[np.ix_(positions, positions)] += 2 * (Xw.transpose() @ X)
        dataRhs[positions] += 2 * (Xw.transpose() @ d['y'])

    return {'freeIndices': freeIndices, 'symMap': symMap, 'dataHessian': dataHessian,
            'dataRhs': dataRhs,
            'penaltyHessian': symMap.transpose() @ symMap if buildSymMapFlag else None}

# End of trappingPrecompute_fn
#--------------------------------------------------------------

def trappingAlternate_fn(pieces, coeffArray, numVars, gamma, nu, maxNumIters, tol):
    """
    Run the trapping relaxation to convergence at one fixed nu.

    Alternates, starting from the unconstrained coefficients:

      (b) auxA <- projection of sym(A) onto {M : M <= gamma * I}, an eigenvalue clamp
          ('symmetricPartProjection_fn'), which is the exact minimizer of the coupling penalty
          over the cone;
      (a) coefficients <- argmin over the active entries of
              sum_v ||X_v b_v - y_v||^2_w  +  (1 / (2 nu)) * ||sym(A(b)) - auxA||_F^2

    Step (a) couples entry (i, j) of the linear block to entry (j, i), and those live in different
    rows, so unlike 'maskedRidgeTowardTarget_fn' it cannot be done row by row -- it is one solve
    over every active entry. The matrix of that solve does not change while nu is fixed, so it is
    factorized once here and only the right-hand side is rebuilt per iteration.

    Parameters
    ----------
    pieces : dict from 'trappingPrecompute_fn'
    coeffArray : np.array of floats, numVars x numFunctionals. The starting point.
    numVars : int
    gamma : float, <= 0
    nu : float, > 0. Smaller nu pulls sym(A) onto the cone harder at more cost to the data fit.
    maxNumIters : int
    tol : float. Convergence tolerance on the largest coefficient change.

    Returns
    -------
    coeffArray : np.array of floats, or None if the solve is singular at this nu
    numEigCalls : int
    convergedFlag : bool
    """
    linearInds = np.arange(1, 1 + numVars)
    symMap = pieces['symMap']
    try:
        factorization = luFactor(pieces['dataHessian'] + (1. / nu) * pieces['penaltyHessian'])
    except (np.linalg.LinAlgError, ValueError):
        return None, 0, False

    numEigCalls = 0
    convergedFlag = False
    for _iteration in range(maxNumIters):
        auxA = symmetricPartProjection_fn(coeffArray[:, linearInds], gamma)
        numEigCalls += 1

        rhs = pieces['dataRhs'] + (1. / nu) * (symMap.transpose() @ auxA.reshape(-1))
        z = luSolve(factorization, rhs)
        if not np.all(np.isfinite(z)):
            return None, numEigCalls, False

        newCoeffArray = np.zeros(coeffArray.shape)
        for k, (v, f) in enumerate(pieces['freeIndices']):
            newCoeffArray[v, f] = z[k]

        change = np.abs(newCoeffArray - coeffArray).max()
        coeffArray = newCoeffArray
        if change < tol:
            convergedFlag = True
            break

    return coeffArray, numEigCalls, convergedFlag

# End of trappingAlternate_fn
#--------------------------------------------------------------

def trappingLinearModel_fn(coeffArray, functionsToUseArray, designMatrices, numVars,
                           boundednessMargin, strictBoundednessMargin, nuLadder, variableNames,
                           outputFilename, normType='spectral', maxNumIters=1000, tol=1e-10,
                           maxNumFreeEntries=2500, innerMargin=None):
    """
    Force a linear model to be bounded using trapping SINDy's relaxation, not the ridge ladder.

    Drop-in alternative to 'stabilizeLinearModel_fn': same arguments in the same order (with
    'nuLadder' in place of 'ridgeLadder') and the same four return values, so callers can switch
    between them. Enforces the same condition and, like the ladder, always returns a model that
    satisfies it.

    METHOD. This is the linear specialization of the scheme in Kaptanoglu et al., Phys. Rev.
    Fluids 6:094401 (2021), Algorithm 1. That method writes the boundedness requirement as
    'sym(shifted linear operator) <= gamma * I' and enforces it by relax-and-split: introduce an
    auxiliary matrix 'auxA' confined to the cone, add a coupling penalty
    ||sym(A) - auxA||^2 / (2 nu), and alternate

        (a) coefficient step -- least squares plus the coupling penalty, see
            'trappingRelaxationStep_fn';
        (b) auxA step -- project sym(A) onto the cone by clamping the eigenvalues of its symmetric
            part at gamma, via 'symmetricPartProjection_fn'.

    Two parts of the published algorithm vanish for a purely linear library, and one is
    strengthened:
      - The energy-preserving constraint on the quadratic coefficients
        (Q_ijk + Q_jik + Q_kij = 0) has nothing to act on.
      - The trapping-region center 'm' drops out of the condition. In the paper the constrained
        quantity is sym(L - Q m), so 'm' is optimized by gradient descent; with Q = 0 the gradient
        is identically zero and sym(A) is all that is left. 'm' still sets the radius of the
        trapping ball, ||c|| / |gamma|, but not whether one exists.
      - The paper's auxA step is a prox-gradient step, auxA <- proj(auxA - alpha (auxA - sym(A))
        / nu). Here the exact minimizer over the cone is available in closed form (it *is* the
        projection of sym(A)), so the alternation uses that instead; it is the same iteration with
        the inner step solved exactly rather than approximately.

    WHY THE nu LADDER, AND WHY IT AIMS PAST THE TARGET. At any fixed nu the relaxation settles
    where sym(A) is close to, but not inside, the cone -- the penalty is finite, so a little
    violation always buys some data fit. Worse, the limit is exactly the boundary: the auxA step
    clamps eigenvalues *at* the margin it is given, so as nu -> 0 the model's max eig sym(A)
    approaches that margin FROM ABOVE and never crosses it. Measured on a planted 20-variable
    system with the A-step aimed at -1e-3: +2.3e-1 at nu = 1e-1, +4.8e-2 at 1e-3, +1.1e-3 at 1e-5,
    -9.95e-4 at 1e-8, -1.000e-3 at 1e-10 -- asymptotic, so a strict 'violation <= -1e-3' test never
    passes at any nu.

    The fix is to aim the A-step at a strictly tighter margin than the one being tested, so the
    boundary it converges to lies inside the required set. 'innerMargin' is that gap: the A-step
    projects onto {M <= (gamma - innerMargin) * I} while acceptance is still tested at gamma. The
    default takes innerMargin = |gamma| (so gamma = -1e-3 is enforced toward -2e-3), which makes
    feasibility reachable early in the anneal instead of never.

    With that in place nu is annealed downward and the first rung whose model is feasible at gamma
    wins. If the whole ladder fails, the fallbacks of 'stabilizeLinearModel_fn' are used: the
    support-preserving projection, then a diagonal shift.

    The anneal is warm-started, each rung beginning from the previous rung's answer. This matters
    for more than speed: the alternation converges slowly (on that same system, nu = 1e-6 from a
    cold start reaches derivative R^2 of -3.6 after 200 iterations and +0.23 after 5000), so a
    cold-started ladder with a modest iteration cap silently reports a far worse model than the
    method can actually produce.

    normType only selects the acceptance test, never the auxA step -- the cone projection is what
    makes the relaxation tractable and it is inherently the 'identity' condition. Since
    sym(A) <= gamma implies max Re eig(A) <= gamma, 'spectral' is satisfied earlier along the
    anneal, so it accepts at a larger nu and costs less data fit. See 'boundednessViolation_fn'.

    COST. The coefficient step is one dense solve of size numFree = the number of active entries,
    so it is O(numFree^3) per iteration where the ladder is O(numVars * activePerRow^3). This is
    the price of the row coupling and it is why 'maxNumFreeEntries' exists: above it the function
    declines to run rather than hang, returning 'skipped-too-large' with the coefficients
    untouched, and the caller should use 'stabilizeLinearModel_fn'.

    Parameters
    ----------
    coeffArray : np.array of floats, numVars x numFunctionals
    functionsToUseArray : np.array of bools, numVars x numFunctionals
    designMatrices : dict, see 'maskedRidgeTowardTarget_fn'. Pass {} to skip to the fallbacks.
    numVars : int
    boundednessMargin : float, <= 0
    strictBoundednessMargin : float, < 0. Used in place of 'boundednessMargin' when that is 0 and
                              the constant functional is active, exactly as in
                              'stabilizeLinearModel_fn'.
    nuLadder : list-like of floats, DECREASING. Smaller nu pulls harder.
    variableNames : list-like of str
    outputFilename : str
    normType : str, 'spectral' or 'identity'
    maxNumIters : int. Iteration cap for the alternation at each nu.
    tol : float. Convergence tolerance on the largest coefficient change.
    maxNumFreeEntries : int. Refuse to run above this many active entries.
    innerMargin : float > 0, or None. How far past gamma the A-step aims; see above. None takes
                  max(|gamma|, 1e-6), which is |gamma| for every nonzero margin the toolkit uses.

    Returns
    -------
    coeffArray : np.array of floats, numVars x numFunctionals
    functionsToUseArray : np.array of bools, numVars x numFunctionals
    actionStr : str
    numEigCalls : int
    """
    linearInds = np.arange(1, 1 + numVars)

    gamma = boundednessMargin
    if gamma == 0 and np.any(functionsToUseArray[:, 0]):
        gamma = strictBoundednessMargin

    # The A-step aims strictly inside the required set, since the relaxation approaches whatever
    # margin it is given from above and would otherwise never satisfy a strict test. See above.
    if innerMargin is None:
        innerMargin = max(abs(gamma), 1e-6)
    gammaInner = gamma - innerMargin

    if functionsToUseArray.shape[1] > 1 + numVars and \
        np.any(functionsToUseArray[:, 1 + numVars:]):
        print('Error: trappingLinearModel_fn was called with active functionals of degree >= 2. ' +
              'The boundedness constraint is implemented for linear models only. Skipping it.')
        return coeffArray, functionsToUseArray, 'skipped-nonlinear-library', 0

    A = coeffArray[:, linearInds]
    violation = boundednessViolation_fn(A, normType)
    numEigCalls = 1
    if violation <= gamma:
        return coeffArray, functionsToUseArray, 'already-bounded', numEigCalls

    numFree = int(np.sum(functionsToUseArray))
    if numFree > maxNumFreeEntries:
        return coeffArray, functionsToUseArray, 'skipped-too-large', numEigCalls

    if len(designMatrices) > 0:
        pieces = trappingPrecompute_fn(coeffArray, functionsToUseArray, designMatrices, numVars)

        # Warm start: each nu begins from the previous nu's answer rather than from the
        # unconstrained fit. This is what makes the anneal a continuation method -- consecutive
        # rungs have nearby solutions, so the alternation needs far fewer iterations, and the
        # iterate stays on the path that tracks the constrained optimum instead of re-converging
        # from scratch to whatever the new nu pulls it to.
        warmStart = coeffArray
        for nu in nuLadder:
            candidateCoeffArray, ne, convergedFlag = trappingAlternate_fn(
                pieces, warmStart, numVars, gammaInner, nu, maxNumIters, tol)
            numEigCalls += ne
            if candidateCoeffArray is None:
                continue
            warmStart = candidateCoeffArray

            numEigCalls += 1
            if boundednessViolation_fn(candidateCoeffArray[:, linearInds], normType) <= gamma:
                return candidateCoeffArray, functionsToUseArray, \
                    'trapping(nu=' + str(nu) + ('' if convergedFlag else ', unconverged') + ')', \
                    numEigCalls

    # Fallbacks, identical to steps 3 and 4 of 'stabilizeLinearModel_fn'.
    functionsToUseArray = functionsToUseArray.copy()
    boundedA, convergedFlag, ne = projectOntoBoundedAndSupport_fn(
        A, functionsToUseArray[:, linearInds], gamma)
    numEigCalls += ne

    if not convergedFlag:
        restoredVars = []
        for v in range(numVars):
            if not functionsToUseArray[v, 1 + v]:
                functionsToUseArray[v, 1 + v] = True
                restoredVars.append(variableNames[v])
        if len(restoredVars) > 0:
            with open(outputFilename, 'a') as file:
                print('Trapping constraint: restored culled self-term(s) for ' +
                      ', '.join(restoredVars) +
                      ', since a bounded model needs a negative diagonal.', file=file)
            boundedA, convergedFlag, ne = projectOntoBoundedAndSupport_fn(
                A, functionsToUseArray[:, linearInds], gamma)
            numEigCalls += ne

    if convergedFlag:
        projectedCoeffArray = coeffArray.copy()
        projectedCoeffArray[:, linearInds] = boundedA
        return projectedCoeffArray, functionsToUseArray, 'projection', numEigCalls

    shift = violation - gamma + 0.01
    coeffArray = coeffArray.copy()
    for v in range(numVars):
        coeffArray[v, 1 + v] = coeffArray[v, 1 + v] - shift
    with open(outputFilename, 'a') as file:
        print('Trapping constraint: nu ladder and projection both failed, applied a ' +
              'diagonal shift of ' + str(np.round(shift, 4)) + ' instead.', file=file)

    return coeffArray, functionsToUseArray, 'diagonalShift', numEigCalls

# End of trappingLinearModel_fn
#--------------------------------------------------------------

def powerIterationNorm_fn(M, numIters=30):
    """
    Estimate the spectral norm of a symmetric positive semidefinite matrix, deterministically.

    The estimate only sets a gradient step size and the scale of the penalty ladder in
    'penaltyLinearModel_fn', so a couple of digits is plenty and being an underestimate is safe
    (the backtracking line search fixes a step that is too long). What it must be is reproducible,
    which is why the start vector is fixed rather than random -- every result in this toolkit is
    checked by re-running under a seed and comparing pickles.

    Parameters
    ----------
    M : np.array of floats, n x n, symmetric positive semidefinite
    numIters : int

    Returns
    -------
    float, an estimate of max eig(M), approached from below.
    """
    numRows = M.shape[0]
    if numRows == 0:
        return 0.
    v = np.cos(np.arange(numRows, dtype=float))  # deterministic and generic, unlike ones()
    vNorm = np.linalg.norm(v)
    if vNorm == 0:
        return 0.
    v = v / vNorm

    estimate = 0.
    for _iteration in range(numIters):
        w = M @ v
        wNorm = float(np.linalg.norm(w))
        if wNorm == 0 or not np.isfinite(wNorm):
            return estimate
        v = w / wNorm
        estimate = wNorm

    return float(estimate)

# End of powerIterationNorm_fn
#--------------------------------------------------------------

def boundednessPenaltyAndGradient_fn(A, gamma, smoothing):
    """
    A smooth measure of HOW unbounded xDot = A*x is, and its gradient with respect to A.

    'boundednessViolation_fn' at normType 'identity' returns max eig sym(A), and the model is
    bounded when that is <= gamma. The quantity to penalize is therefore the excess,
    max(0, max eig sym(A) - gamma), which is convex in A but not differentiable -- it kinks
    wherever the top eigenvalue is repeated or the constraint becomes active, which is exactly
    where an optimizer spends its time. This returns the softplus-of-eigenvalues smoothing of it,

        phi(A) = smoothing * log(1 + sum_i exp((lambda_i - gamma) / smoothing)),

    lambda_i the eigenvalues of sym(A). Three properties are what make it usable here:

      1. It is a strict UPPER bound on the true excess, and within smoothing * log(1 + numVars) of
         it. So driving phi down drives the real violation down with it, and any error is on the
         conservative side -- the opposite of the trapping relaxation, whose iterate approaches the
         margin from ABOVE and so needs 'innerMargin' to aim past the target (see
         'trappingLinearModel_fn').
      2. It is convex (a symmetric convex spectral function of sym(A), composed with a linear map),
         so the penalized regression below has no local minima to get stuck in.
      3. Its gradient is bounded: ||grad||_F <= 1, with equality only deep inside the violating
         region. A penalty with a bounded gradient is an EXACT penalty -- past a finite weight the
         data term cannot buy any more violation -- which is why a finite rung of the ladder lands
         strictly inside the feasible set instead of converging to its boundary.

    The gradient with respect to A, not sym(A): for a spectral function f(S) = sum_i g(lambda_i),
    grad_S f = V diag(g'(lambda)) V', and since sym(A)[i, j] = (A[i, j] + A[j, i]) / 2 and that
    gradient is symmetric, the chain rule gives grad_A f = grad_S f unchanged.

    Parameters
    ----------
    A : np.array of floats, numVars x numVars
    gamma : float, <= 0. The margin the penalty measures the excess over.
    smoothing : float, > 0. Larger is smoother, more conservative, and better conditioned; the
                gradient of phi is Lipschitz with constant <= 1 / smoothing.

    Returns
    -------
    value : float, >= 0
    gradient : np.array of floats, numVars x numVars, symmetric
    """
    eigVals, eigVecs = np.linalg.eigh((A + A.transpose()) / 2)
    exponents = (eigVals - gamma) / smoothing

    # Shift before exponentiating, so that neither a large violation nor a large margin overflows.
    # The '1 +' in the definition is the 0 of max(0, .), hence the exp(-shift) term:
    shift = max(0., float(exponents.max()))
    expTerms = np.exp(exponents - shift)
    denominator = float(np.exp(-shift) + expTerms.sum())
    value = smoothing * (shift + np.log(denominator))
    gradient = (eigVecs * (expTerms / denominator)) @ eigVecs.transpose()

    return value, gradient

# End of boundednessPenaltyAndGradient_fn
#--------------------------------------------------------------

def penaltyPrecompute_fn(pieces, numVars):
    """
    Index and scaling arrays that 'penaltyMinimize_fn' needs on top of 'trappingPrecompute_fn'.

    Two jobs. First, the maps between the flat vector of free coefficients and the matrix A, so the
    penalty can be evaluated and its gradient scattered back without going through 'symMap'.
    Second, a Jacobi rescaling of the free coordinates: the solve is iterative rather than direct,
    so unlike the trapping step it pays for the conditioning of the data Hessian, and on a library
    of functionals with wildly different magnitudes (the loyalty series being the case in point)
    the diagonal alone spans several orders of magnitude. Working in the variables
    zeta = sqrt(diag(dataHessian)) * z gives the data Hessian a unit diagonal and costs nothing.

    Parameters
    ----------
    pieces : dict from 'trappingPrecompute_fn'
    numVars : int

    Returns
    -------
    dict with keys:
      'scale'           : np.array of floats, numFree. sqrt of diag(dataHessian), floored at 1 where
                          a functional carries no curvature at all.
      'scaledHessian'   : np.array of floats, numFree x numFree, unit diagonal
      'scaledRhs'       : np.array of floats, numFree
      'freeVarInds'     : np.array of ints, numFree. Variable of each free entry.
      'freeFnInds'      : np.array of ints, numFree. Functional of each free entry.
      'isLinearEntry'   : np.array of bools, numFree. Which free entries live in the block A, ie
                          are degree-1 monomials. The constant is excluded, so the affine term is
                          untouched by the penalty, exactly as 'symMap' arranges for trapping.
      'linearRowInds'   : np.array of ints. Row of A of each linear entry.
      'linearColInds'   : np.array of ints. Column of A of each linear entry.
      'dataNorm'        : float, an estimate of ||dataHessian||_2
      'scaledDataNorm'  : float, an estimate of ||scaledHessian||_2
    """
    freeIndices = pieces['freeIndices']
    freeVarInds = np.array([v for v, _f in freeIndices], dtype=int)
    freeFnInds = np.array([f for _v, f in freeIndices], dtype=int)
    isLinearEntry = (freeFnInds >= 1) & (freeFnInds <= numVars)

    dataHessian = pieces['dataHessian']
    scale = np.sqrt(np.maximum(np.diag(dataHessian), 0.))
    scale[scale <= 0] = 1.  # a functional whose design column is identically zero

    return {'scale': scale,
            'scaledHessian': dataHessian / np.outer(scale, scale),
            'scaledRhs': pieces['dataRhs'] / scale,
            'freeVarInds': freeVarInds,
            'freeFnInds': freeFnInds,
            'isLinearEntry': isLinearEntry,
            'linearRowInds': freeVarInds[isLinearEntry],
            'linearColInds': freeFnInds[isLinearEntry] - 1,
            'dataNorm': powerIterationNorm_fn(dataHessian),
            'scaledDataNorm': powerIterationNorm_fn(dataHessian / np.outer(scale, scale))}

# End of penaltyPrecompute_fn
#--------------------------------------------------------------

def penaltyMinimize_fn(solvePieces, coeffArray, numVars, gamma, lam, smoothing, maxNumIters, tol,
                       maxNumBacktracks=40):
    """
    Minimize 'weighted least squares + lam * unboundedness of A' over the active entries.

    The objective, in the free coefficients z (see 'penaltyPrecompute_fn' for the rescaling that
    the iteration actually runs in):

        sum_v ||X_v b_v - y_v||^2_w  +  lam * phi(A(z)),

    phi from 'boundednessPenaltyAndGradient_fn'. Both terms are convex and differentiable, so this
    is solved by accelerated gradient descent (FISTA, Beck & Teboulle 2009) with a backtracking
    line search and the gradient restart of O'Donoghue & Candes 2015 -- momentum is dropped
    whenever it starts pushing uphill, which matters here because the penalty's curvature changes
    abruptly as eigenvalues cross the margin.

    WHY NOT A DIRECT SOLVE, as the trapping step uses. The penalty couples entry (i, j) of A to
    entry (j, i) just as the trapping coupling term does, but it is not quadratic, so there is no
    fixed matrix to factorize once. The trapping alternation gets one by freezing the auxiliary
    matrix, which is what turns its penalty into a quadratic -- and that is precisely the choice
    that costs it exact feasibility. Paying an iterative solve is what buys the exact penalty.

    The step size is adaptive in both directions: doubled on a failed line search, shrunk by 20 %
    after every successful iteration, so an overestimated starting Lipschitz constant costs a few
    iterations rather than the whole budget.

    Parameters
    ----------
    solvePieces : dict from 'penaltyPrecompute_fn', with the keys of 'trappingPrecompute_fn' merged
                  in ('freeIndices' is not needed, the index arrays replace it)
    coeffArray : np.array of floats, numVars x numFunctionals. The starting point, ie the warm start.
    numVars : int
    gamma : float, <= 0. The margin the penalty measures the excess over.
    lam : float, > 0. Weight of the penalty. Larger pulls harder at more cost to the data fit.
    smoothing : float, > 0. See 'boundednessPenaltyAndGradient_fn'.
    maxNumIters : int
    tol : float. Convergence tolerance on the largest coefficient change, in unscaled units, so it
          means the same thing as the tolerance of 'trappingAlternate_fn'.
    maxNumBacktracks : int

    Returns
    -------
    coeffArray : np.array of floats, or None if the iteration left the finite numbers
    numEigCalls : int
    convergedFlag : bool
    """
    scale = solvePieces['scale']
    scaledHessian = solvePieces['scaledHessian']
    scaledRhs = solvePieces['scaledRhs']
    isLinearEntry = solvePieces['isLinearEntry']
    linearRowInds = solvePieces['linearRowInds']
    linearColInds = solvePieces['linearColInds']
    freeVarInds = solvePieces['freeVarInds']
    freeFnInds = solvePieces['freeFnInds']

    def objectiveAndGradient_fn(zeta):
        """Value and gradient at a point, in the rescaled coordinates."""
        z = zeta / scale
        A = np.zeros((numVars, numVars))
        A[linearRowInds, linearColInds] = z[isLinearEntry]
        penaltyValue, penaltyGradient = boundednessPenaltyAndGradient_fn(A, gamma, smoothing)

        hessianTimesZeta = scaledHessian @ zeta
        value = 0.5 * float(zeta @ hessianTimesZeta) - float(scaledRhs @ zeta) + lam * penaltyValue

        gradient = hessianTimesZeta - scaledRhs
        gradientOfPenalty = np.zeros(len(zeta))
        gradientOfPenalty[isLinearEntry] = penaltyGradient[linearRowInds, linearColInds]

        return value, gradient + lam * gradientOfPenalty / scale

    x = scale * coeffArray[freeVarInds, freeFnInds]
    xPrevious = x.copy()
    momentumWeight = 1.
    # Both terms are upper bounds on the true curvature, so the line search should mostly be
    # shrinking this rather than growing it:
    lipschitz = max(solvePieces['scaledDataNorm'] + lam / smoothing, 1e-12)

    numEigCalls = 0
    convergedFlag = False
    for _iteration in range(maxNumIters):
        nextMomentumWeight = (1. + np.sqrt(1. + 4. * momentumWeight * momentumWeight)) / 2.
        y = x + ((momentumWeight - 1.) / nextMomentumWeight) * (x - xPrevious)

        valueAtY, gradientAtY = objectiveAndGradient_fn(y)
        numEigCalls += 1
        if not np.isfinite(valueAtY) or not np.all(np.isfinite(gradientAtY)):
            return None, numEigCalls, False

        for _backtrack in range(maxNumBacktracks):
            xNew = y - gradientAtY / lipschitz
            valueAtNew, _gradientAtNew = objectiveAndGradient_fn(xNew)
            numEigCalls += 1
            step = xNew - y
            quadraticModel = valueAtY + float(gradientAtY @ step) + \
                0.5 * lipschitz * float(step @ step)
            if np.isfinite(valueAtNew) and valueAtNew <= quadraticModel + 1e-12 * abs(valueAtY):
                break
            lipschitz = lipschitz * 2.

        if not np.all(np.isfinite(xNew)):
            return None, numEigCalls, False

        change = float(np.abs((xNew - x) / scale).max())
        # Gradient restart: the momentum has overshot if the step it produced disagrees with the
        # descent direction. Cheaper and more robust than restarting on the function value.
        restartFlag = float(gradientAtY @ (xNew - x)) > 0
        xPrevious = x
        x = xNew
        momentumWeight = 1. if restartFlag else nextMomentumWeight
        lipschitz = max(lipschitz * 0.8, 1e-12)

        if change < tol:
            convergedFlag = True
            break

    newCoeffArray = np.zeros(coeffArray.shape)
    newCoeffArray[freeVarInds, freeFnInds] = x / scale

    return newCoeffArray, numEigCalls, convergedFlag

# End of penaltyMinimize_fn
#--------------------------------------------------------------

def penaltyLinearModel_fn(coeffArray, functionsToUseArray, designMatrices, numVars,
                          boundednessMargin, strictBoundednessMargin, penaltyLadder, variableNames,
                          outputFilename, normType='spectral', maxNumIters=1000, tol=1e-10,
                          maxNumFreeEntries=2500, smoothingFactor=0.25, innerMargin=0.,
                          numPolishPasses=2):
    """
    Force a linear model to be bounded by putting the unboundedness measure INTO the regression loss.

    Drop-in alternative to 'stabilizeLinearModel_fn' and 'trappingLinearModel_fn': same arguments in
    the same order (with 'penaltyLadder' in place of 'ridgeLadder' / 'nuLadder') and the same four
    return values. Enforces the same condition and, like both of those, always returns a model that
    satisfies it.

    METHOD, AND HOW IT DIFFERS FROM THE OTHER TWO. All three end at a feasible model; what differs
    is what the regression is actually asked to minimize.

      ladder    The constraint never enters the loss. A stabilized TARGET matrix is computed once
                from the current A and frozen, and the loss becomes least squares plus
                lam * ||b - target||^2. So the regression is pulled toward one particular bounded
                matrix, chosen before the refit and never revisited, which is why it is blunt: it
                moves every coefficient toward that matrix whether or not doing so helps
                feasibility.
      trapping  The constraint enters as a quadratic coupling to an auxiliary matrix confined to
                the cone, ||sym(A) - auxA||^2 / 2nu, with auxA re-projected from the CURRENT A each
                iteration. The target moves, which is the improvement over the ladder, but the
                penalty is still a squared distance, and a squared distance has zero gradient at
                the boundary: the iterate approaches the margin asymptotically from OUTSIDE and
                never crosses it, so nu has to be annealed and the A-step aimed past the target
                ('innerMargin') for a strict test to ever pass.
      penalty   The loss is least squares plus lam * phi(A), phi being the smoothed excess
                max(0, max eig sym(A) - gamma) itself (see 'boundednessPenaltyAndGradient_fn') --
                the same quantity 'boundednessViolation_fn' reports and the same one the acceptance
                test reads. No auxiliary matrix, no frozen target: the thing being minimized IS the
                thing being tested.

    WHY THAT CHANGES THE OUTCOME AND NOT JUST THE BOOKKEEPING. phi has a bounded gradient
    (||grad||_F <= 1), so it is an EXACT penalty: past a finite lam the data term can no longer buy
    any violation, and the minimizer is strictly feasible rather than asymptotically so. At the
    optimum the two forces balance, lam * ||grad phi|| = ||data gradient||, and working that through
    the softplus gives

        max eig sym(A) - gamma  ~  smoothing * log(||data gradient|| / lam),

    ie feasibility from the moment lam passes the scale of the data gradient, with the excess
    conservatism past that growing only logarithmically. Contrast the quadratic penalty, where the
    same balance gives a violation proportional to nu and hence a strict test that never passes.
    That is also why 'innerMargin' defaults to 0 here where 'trappingLinearModel_fn' needs |gamma|:
    phi is an upper bound on the true excess, so the smoothing error already points into the
    feasible set.

    THE POLISH PASSES ARE NOT OPTIONAL POLISH. The same balance says the accepted model sits
    'smoothing' * log(...) INSIDE the margin, and 'smoothing' at the winning rung is set from the
    PREVIOUS rung's violation, which is still outside it -- so the first feasible rung typically
    overshoots by a factor of 100, and every bit of that overshoot is data fit spent on
    conservatism nobody asked for (measured on a planted 8-variable system: max eig sym(A) of
    -1.1e-1 against a gamma of -1e-3). A polish pass re-solves at the SAME lam with the smoothing
    re-derived from the now-feasible iterate, where the floor at |gamma| applies, so the model
    relaxes back out toward the margin under the data term while staying inside it. The passes are
    warm-started and converge immediately (the floor makes the second one a no-op), and a pass
    whose result is not feasible is discarded rather than accepted.

    THE LADDER IS SCALED TO THE PROBLEM, not absolute. lam trades a squared residual against an
    eigenvalue, so unlike the ridge ladder (coefficients against coefficients) a fixed value means
    nothing across datasets. The balance above says the interesting scale is
    ||dataHessian||_2 * (violation - gamma), so 'penaltyLadder' is a list of MULTIPLES of that, and
    a rung near 1 is where feasibility is expected to appear. Rungs are warm-started, and the
    smoothing is re-derived from the warm start's own violation at each rung, so the penalty
    sharpens exactly as fast as the iterate approaches the margin -- keeping (violation - gamma) /
    smoothing at 1 / 'smoothingFactor' throughout, which is what stops the conditioning from
    running away as lam grows.

    normType selects only the acceptance test, never the penalty, which is inherently the
    'identity' condition on sym(A) -- the same arrangement as 'trappingLinearModel_fn'. Since
    sym(A) <= gamma*I implies max Re eig(A) <= gamma, 'spectral' is satisfied at an earlier rung
    and so costs less data fit.

    COST. One dense matrix-vector product of size numFree plus one numVars eigendecomposition per
    line-search evaluation, and a numFree x numFree Hessian is built up front, so 'maxNumFreeEntries'
    applies exactly as it does for trapping: above it this returns 'skipped-too-large' with the
    coefficients untouched and the caller should use 'stabilizeLinearModel_fn'.

    Parameters
    ----------
    coeffArray : np.array of floats, numVars x numFunctionals
    functionsToUseArray : np.array of bools, numVars x numFunctionals
    designMatrices : dict, see 'maskedRidgeTowardTarget_fn'. Pass {} to skip to the fallbacks.
    numVars : int
    boundednessMargin : float, <= 0
    strictBoundednessMargin : float, < 0. Used in place of 'boundednessMargin' when that is 0 and
                              the constant functional is active, exactly as in the other two.
    penaltyLadder : list-like of floats, INCREASING. Multiples of the automatic scale; see above.
    variableNames : list-like of str
    outputFilename : str
    normType : str, 'spectral' or 'identity'
    maxNumIters : int. Iteration cap for the accelerated gradient solve at each rung.
    tol : float. Convergence tolerance on the largest coefficient change.
    maxNumFreeEntries : int. Refuse to run above this many active entries.
    smoothingFactor : float, > 0. The smoothing of phi, as a fraction of the current violation gap.
                      Larger is better conditioned and more conservative; the accepted model sits
                      roughly smoothingFactor * |gamma| * log(lam ratio) inside the margin.
    innerMargin : float, >= 0. How far past gamma the penalty aims. Unlike trapping this can be 0,
                  see above; it is exposed for symmetry and for tightening if a run needs it.
    numPolishPasses : int, >= 0. Extra solves at the winning lam with the smoothing re-derived from
                      the feasible iterate. See above; 0 disables.

    Returns
    -------
    coeffArray : np.array of floats, numVars x numFunctionals
    functionsToUseArray : np.array of bools, numVars x numFunctionals
    actionStr : str
    numEigCalls : int
    """
    linearInds = np.arange(1, 1 + numVars)

    gamma = boundednessMargin
    if gamma == 0 and np.any(functionsToUseArray[:, 0]):
        gamma = strictBoundednessMargin
    gammaInner = gamma - innerMargin

    if functionsToUseArray.shape[1] > 1 + numVars and \
        np.any(functionsToUseArray[:, 1 + numVars:]):
        print('Error: penaltyLinearModel_fn was called with active functionals of degree >= 2. ' +
              'The boundedness constraint is implemented for linear models only. Skipping it.')
        return coeffArray, functionsToUseArray, 'skipped-nonlinear-library', 0

    A = coeffArray[:, linearInds]
    violation = boundednessViolation_fn(A, normType)
    numEigCalls = 1
    if violation <= gamma:
        return coeffArray, functionsToUseArray, 'already-bounded', numEigCalls

    numFree = int(np.sum(functionsToUseArray))
    if numFree > maxNumFreeEntries:
        return coeffArray, functionsToUseArray, 'skipped-too-large', numEigCalls

    if len(designMatrices) > 0:
        pieces = trappingPrecompute_fn(coeffArray, functionsToUseArray, designMatrices, numVars,
                                       buildSymMapFlag=False)
        solvePieces = penaltyPrecompute_fn(pieces, numVars)

        # The penalty is on sym(A) whatever the acceptance test is, so the gap that sets the scale
        # of both lam and the smoothing is the 'identity' one, not the tested one:
        violationForScale = boundednessViolation_fn(A, 'identity')
        numEigCalls += 1
        gapForScale = max(violationForScale - gammaInner, abs(gamma), 1e-12)
        lamScale = max(solvePieces['dataNorm'], 1e-12) * gapForScale

        warmStart = coeffArray
        for rung in penaltyLadder:
            lam = rung * lamScale
            # Re-derived per rung from the warm start, so the penalty sharpens only as fast as the
            # iterate closes on the margin. See the docstring.
            currentViolation = boundednessViolation_fn(warmStart[:, linearInds], 'identity')
            numEigCalls += 1
            smoothing = smoothingFactor * max(currentViolation - gammaInner, abs(gamma), 1e-12)

            candidateCoeffArray, ne, convergedFlag = penaltyMinimize_fn(
                solvePieces, warmStart, numVars, gammaInner, lam, smoothing, maxNumIters, tol)
            numEigCalls += ne
            if candidateCoeffArray is None:
                continue
            warmStart = candidateCoeffArray

            numEigCalls += 1
            if boundednessViolation_fn(candidateCoeffArray[:, linearInds], normType) > gamma:
                continue

            # Feasible. Now walk the overshoot back in, see the docstring:
            numPolished = 0
            for _polishPass in range(numPolishPasses):
                polishViolation = boundednessViolation_fn(candidateCoeffArray[:, linearInds],
                                                          'identity')
                numEigCalls += 1
                polishSmoothing = smoothingFactor * max(polishViolation - gammaInner, abs(gamma),
                                                        1e-12)
                if polishSmoothing >= smoothing:  # nothing left to sharpen
                    break
                polishedCoeffArray, ne, polishedConvergedFlag = penaltyMinimize_fn(
                    solvePieces, candidateCoeffArray, numVars, gammaInner, lam, polishSmoothing,
                    maxNumIters, tol)
                numEigCalls += ne
                if polishedCoeffArray is None:
                    break
                numEigCalls += 1
                if boundednessViolation_fn(polishedCoeffArray[:, linearInds], normType) > gamma:
                    break  # the relaxation went too far, keep the last feasible model
                candidateCoeffArray = polishedCoeffArray
                convergedFlag = polishedConvergedFlag
                smoothing = polishSmoothing
                numPolished += 1

            return candidateCoeffArray, functionsToUseArray, \
                'penalty(lam=' + str(np.format_float_scientific(lam, precision=2)) + \
                (', polished x%d' % numPolished if numPolished else '') + \
                ('' if convergedFlag else ', unconverged') + ')', numEigCalls

    # Fallbacks, identical to steps 3 and 4 of 'stabilizeLinearModel_fn'.
    functionsToUseArray = functionsToUseArray.copy()
    boundedA, convergedFlag, ne = projectOntoBoundedAndSupport_fn(
        A, functionsToUseArray[:, linearInds], gamma)
    numEigCalls += ne

    if not convergedFlag:
        restoredVars = []
        for v in range(numVars):
            if not functionsToUseArray[v, 1 + v]:
                functionsToUseArray[v, 1 + v] = True
                restoredVars.append(variableNames[v])
        if len(restoredVars) > 0:
            with open(outputFilename, 'a') as file:
                print('Penalty constraint: restored culled self-term(s) for ' +
                      ', '.join(restoredVars) +
                      ', since a bounded model needs a negative diagonal.', file=file)
            boundedA, convergedFlag, ne = projectOntoBoundedAndSupport_fn(
                A, functionsToUseArray[:, linearInds], gamma)
            numEigCalls += ne

    if convergedFlag:
        projectedCoeffArray = coeffArray.copy()
        projectedCoeffArray[:, linearInds] = boundedA
        return projectedCoeffArray, functionsToUseArray, 'projection', numEigCalls

    shift = violation - gamma + 0.01
    coeffArray = coeffArray.copy()
    for v in range(numVars):
        coeffArray[v, 1 + v] = coeffArray[v, 1 + v] - shift
    with open(outputFilename, 'a') as file:
        print('Penalty constraint: penalty ladder and projection both failed, applied a ' +
              'diagonal shift of ' + str(np.round(shift, 4)) + ' instead.', file=file)

    return coeffArray, functionsToUseArray, 'diagonalShift', numEigCalls

# End of penaltyLinearModel_fn
#--------------------------------------------------------------

def findSpansOfFunctionals_fn(fnArray, fnVals, fnValsNoisy, wts, includeConstantFlag, p):
    """
    Find which functionals are in the linear span of the final selected functionals (per variable).
    Return R-squared values for each {functional, variable} pair (each variable has a few 
    selected functionals).

    Parameters
    ----------
    fnArray : np.array of booleans, numVars x numFunctionals in library. Shows which functionals
              were used for each variable.
    fnVals : np.array, numTimepoints (used) x numFunctionals. Gives values of functionals at 
             timepoints, using the smoothed time-series
    fnValsNoisy : np.array, numTimepoints (used) x numFunctionals. Gives values of functionals at 
             timepoints, but using the post-added noise, pre-smoothing time-series. 
    wts : np.array, numTimepoints (used) x numVars. Gives weights of timepoints 
    includeConstantFlag : bool 
    p : dict of miscellaneous parameters, including:
        functionList : list of str
        variableNames : list of str
        outputFilename : str
        plottingThreshold : float
        windowLengthForFomEnvelopes : int
        dt : float
        margin : int
        maxFnValRatio : float
        minNumStartIndsToUse : int

    Returns
    -------
    rSq : np.array of floats, size fnArray.shape. Gives the R-squared from linear fits of
                each functional using just the selected functionals (per variable).
    betaZeros : np.array of floats, size fnArray.shape. Gives .intercept_ of each fit
    betas : np.array of lists, array has size fnArray.shape. Gives the .coef_ of each fit

    """

    betaZeros = np.zeros(fnArray.shape)  # to save the intercepts
    betas = np.empty(fnArray.shape, dtype=object)  # to save the coefficients
    rSq = np.zeros(fnArray.shape)  # to save R-squared values
    lrSp = LinearRegression(fit_intercept=includeConstantFlag)
    for v in range(fnArray.shape[0]):
        if np.sum(fnArray[v, :]) == 0:
            console = sys.stdout
            with open(p['outputFilename'], 'a') as file:
                print(p['variableNames'][v] + ' has no functionals left in sparse library.', 
                      file=file)
                sys.stdout = console
                file.close()
        else:  # case: there are functionals in this variable's sparse library
            basis = np.where(fnArray[v, :])[0]
            basisStr = str(np.array(p['functionList'])[basis]).replace('[','').replace(']','')
            X = fnVals[:, fnArray[v, :]]  # The values of the selected functionals
            for i in range(fnArray.shape[1]):  # num functionals
                if fnArray[v, i]:
                    rSq[v, i] = 1
                else:
                    y = fnVals[:, i]
                    # Exclude any timepoints from the regression with very large differences in
                    # functional values used as features: 
                    miniFnArray = np.ones((1, X.shape[1]), dtype=bool)
                    removeMarginFlag = False # since for FoM timepoint assessment, we don't ignore 
                    # a margin at each end.
                    nonConstantIndices = np.where(np.array(p['functionList']) != '1')[0]
                    goodInds = \
                        parseTimepointsByMagnitudesOfVariables_fn(X.copy(), miniFnArray, 
                                                                  nonConstantIndices, p['margin'], 
                                                                  p['maxFnValRatio'], 
                                                                  p['minNumStartIndsToUse'],
                                                                  removeMarginFlag)
                    # goodInds is 1 x numTimepoints vector
                    theseWts = wts[:, v].copy() 
                    theseWts[np.logical_not(goodInds.flatten())] = 0 
                    # theseWts = theseWts / np.sum(goodInds)  # normalize the sum
                    lrSp = lrSp.fit(X, y, sample_weight=theseWts)
                    rSq[v, i] = lrSp.score(X, y, sample_weight=theseWts)
                    betaZeros[v, i] = lrSp.intercept_
                    betas[v, i] = lrSp.coef_
                    # plot functionals that are somewhat well-fit:
                    showNoiseEnvelopesFlag = True
                    if rSq[v, i] > p['plottingThreshold'] and rSq[v, i] < 1:  
                        # < 1 to ignore basis functionals
                        yHat = lrSp.predict(X)
                        plt.figure()
                        plt.title('var = ' + p['variableNames'][v] + ', basis = ' + \
                                  basisStr + '\n' + 'functional = ' + \
                                  p['functionList'][i] + ', R_sq = ' + str(np.round(rSq[v, i], 2)),
                                  fontsize=14, fontweight='bold')
                        plt.scatter(np.array(range(len(y))), y, s=3, c='k', 
                                    label='functional values')
                        plt.scatter(np.array(range(len(y))), yHat, s=3, c='r', 
                                    label='fit values')
                        if showNoiseEnvelopesFlag:
                            slopeY, stdY, meanY = \
                                calculateSlopeAndStd_fn(fnValsNoisy[:, i], p['dt'],
                                                        p['windowLengthForFomEnvelopes'])
                            plt.plot(meanY + 2 * stdY, 'g', label='two std dev of noisy data')
                            plt.plot(meanY - 2 * stdY, 'g')
                        plt.xlabel('FoM timepoints', fontsize=14, fontweight='bold')
                        plt.ylabel('functional value', fontsize=14, fontweight='bold')
                        plt.legend(fontsize=14)

    return rSq, betaZeros, betas

# End of findSpansOfFunctionals_fn
#---------------------------------------------------------------------

def findSpansOfFunctionalsLeaveOneOut_fn(fnArray, fnVals, fnValsNoisy, wts, includeConstantFlag, 
                                         p):
    """
    Considering the retained functionals (per variable), find which functionals are in the linear 
    span of the other functionals (leave-one-out).
    Return rR-squared values for each {functional, variable} pair (each variable has a few 
    selected functionals).

    Parameters
    ----------
    fnArray : np.array of booleans, numVars x numFunctionals in library. Shows which functionals
              are active for each variable.
    fnVals : np.array, numTimepoints (used) x numFunctionals. Gives values of functionals at 
             timepoints, using the post-smoothed time-series.
    fnValsNoisy : np.array, numTimepoints (used) x numFunctionals. Gives values of functionals at 
             timepoints, using the post-added noise and pre-smoothing time-series. 
    wts : np.array, numTimepoints (used) x numVars. Gives weights of timepoints 
    includeConstantFlag : bool 
    p : dict of miscellaneous parameters, including:
        functionList : list of str
        variableNames : list of str
        outputFilename : str
        windowLengthForFomEnvelopes : int
        dt : float
        margin : int
        maxFnValRatio : float
        minNumStartIndsToUse : int
        plottingThreshold : float

    Returns
    -------
    rSq : np.array of floats, size fnArray.shape. Gives the R-squared from linear fits of
                each functional using just the selected functionals (per variable).
    betaZeros : np.array of floats, size fnArray.shape. The {i,j}th entry is the intercept for the 
                fit of the j'th functional using the active functionals, for the i'th variable.
    betas : np.array of objects, size fnArray.shape. The {i,j}th object is the vector of beta 
            coefficients found by fitting the j'th functional using the active functionals, for 
            the i'th variable. If the j'th functional is not active, the object = []. Else its
            length = # active functionals for the variable.

    """  

    betaZeros = np.zeros(fnArray.shape)  # to save .intercept_
    betas = np.empty(fnArray.shape, dtype=object)  # to save .coef_
    rSq = np.zeros(fnArray.shape)
    functionListArray = np.array(p['functionList'])  # hoisted out of the leave-one-out loops below
    nonConstantIndices = np.where(functionListArray != '1')[0]
    for v in range(fnArray.shape[0]):  # loop over numVars
        if np.sum(fnArray[v, :]) <= 1:
            if p.get('outputFilename', ''):
                console = sys.stdout
                with open(p['outputFilename'], 'a') as file:
                    print(p['variableNames'][v] + ' has <= 1 functional left in library.', file=file)
                    sys.stdout = console
                    file.close()
        else:  # case: there are functionals in this variable's sparse library
            inds = np.where(fnArray[v, :]==True)[0]
            # Constant for every leave-one-out fit of this variable, so set them up once here:
            miniFnArray = np.ones((1, len(inds) - 1))
            removeMarginFlag = False # since for FoM timepoint assessment, we don't ignore a
            # margin at each end.
            for i in inds:  # num functionals
                y = fnVals[:, i]
                others =  inds[inds != i]  # already a fresh array, no copy needed
                X = fnVals[:, others]  # the other retained functionals (fancy indexing copies)
                # Exclude any timepoints from the regression with very large differences in
                # functional values used as features:
                goodInds = \
                    parseTimepointsByMagnitudesOfVariables_fn(X, miniFnArray,
                                                              nonConstantIndices,
                                                              p['margin'], p['maxFnValRatio'],
                                                              p['minNumStartIndsToUse'],
                                                              removeMarginFlag)
                # goodInds is 1 x numTimepoints vector
                theseWts = wts[:, v].copy()
                theseWts[np.logical_not(goodInds.flatten())] = 0
                theseWts = theseWts / np.sum(theseWts) # normalize back to sum = 1
                # Fit, then score, without going through scikit-learn's per-call validation (which
                # costs more than these small solves). 'yHat' and 'rSq' below reproduce
                # LinearRegression.predict() and .score() (ie sklearn's weighted r2_score):
                theseCoeffs, thisIntercept = \
                    weightedLeastSquares_fn(X, y, theseWts, includeConstantFlag)
                yHat = X @ theseCoeffs + thisIntercept
                numerator = np.sum(theseWts * (y - yHat) ** 2)
                denominator = np.sum(theseWts * (y - np.average(y, weights=theseWts)) ** 2)
                rSq[v, i] = 1 - numerator / denominator
                betaZeros[v, i] = thisIntercept
                betas[v, i] = theseCoeffs
                # plot functionals that are somewhat well-fit:
                showNoiseEnvelopesFlag = True
                if rSq[v, i] > p['plottingThreshold'] and rSq[v, i] < 1:  # < 1 to ignore basis
                # functionals, which have rSq = 1.
                    basisStr = \
                        str(functionListArray[others]).replace('[','').replace(']','')  # only
                        # needed for the plot title, so build it here rather than every iteration.
                    plt.figure()  # 'yHat' is already the fitted values, computed above.
                    plt.title('var = ' + p['variableNames'][v] + ', sub-basis = ' + \
                              basisStr + '\n' + 'left-out functional = ' + \
                              p['functionList'][i] + ', R_sq = ' + str(np.round(rSq[v, i], 2)),
                              fontsize=14, fontweight='bold')
                    plt.scatter(np.array(range(len(y))), y, s=3, c='k', 
                                label='functional values')
                    plt.scatter(np.array(range(len(y))), yHat, s=3, c='r', 
                                label='fit values')
                    if showNoiseEnvelopesFlag:
                        slopeY, stdY, meanY = \
                            calculateSlopeAndStd_fn(fnValsNoisy[:, i], p['dt'],
                                                    p['windowLengthForFomEnvelopes'])
                        plt.plot(meanY + 2 * stdY, 'g', label='two std dev of noisy data')
                        plt.plot(meanY - 2 * stdY, 'g')
                    plt.xlabel('FoM timepoints', fontsize=14, fontweight='bold')
                    plt.ylabel('functional value', fontsize=14, fontweight='bold')
                    plt.legend(fontsize=14)

    return rSq, betaZeros, betas

# End of findSpansOfFunctionalsLeaveOneOut_fn
#----------------------------------------------------------

def findClosestLinearlyEquivalentVersion_fn(nC, tC, rSqLooV, betasLooV, params):
    """
    Given a set of coefficients (for one variable), use linear combinations of true 
    functionals, whose leave-one-out linear fits have sufficiently high Rsquared values, to 
    iteratively substitute to minimize the maximum absolute error in coefficients of true 
    functionals relative to the true system coefficients. 
    We only substitute using linear combinations among true functionals. 
    Note that any functional in the discovered model that is not part of the true functionals has
    no error measure.
    In this function, the inputs are for one system variable (e.g. 'x', 'y').

    Parameters
    ----------
    nCR : np.array (vector) of floats. The model coefficients 
    tCR : np.array (vector) of floats. The true system coefficients 
    rSqLooV : np.array (vector) of floats. The Rsquared values for leave-one-out linear fits
    betasLooV : np.array of objects. numVariables x numLibraryFunctionals. The {i,j}th object is 
                the vector of beta coefficients found by fitting the j'th functional using the 
                active functionals, for the i'th variable. If the j'th functional is not active, 
                the object = []. Else its length = # active functionals for the variable.
    params : dict of miscellaneous parameters

    Returns
    -------
    nC : np.array (vector) of floats. These are coefficients for a model that is linearly 
         equivalent to the argin model 'nC' and with a minimal maximum error relative to true 
         functional coefficients.
    coeffErrR : np.array (vector) of floats. The final percentage absolute coefficient errors of
                each true library functional. length = number of true functionals, ordered 
                according to how the functionals are listed in params['functionList'].

    """
    # Method: Iteratively, do a transform that reduces the biggest error. 
    tF = tC.astype(bool)
    nCR = nC[tF]  # the 'R' means Restricted, to just true functionals. We modify these only.
    tCR = tC[tF]  # ditto 
    numActiveFnals = np.sum(nCR != 0)  # will be used to detect resurrected fnals.
    resurrectedTrueFnalFlag = False
    # We also need the full index of each fnal:
    tC2 = tC.copy()
    tC2[tC2 == 0] = 1  # since this will be in a denominator in loop below
    # initialize for while loop:
    coeffErrR = np.abs((nCR - tCR) / tCR)
    newErr = max(coeffErrR[np.abs(nCR) > 0]) # don't consider error in missing fnals, 
    # which have error = -100% 
    oldErr = 2 * newErr 
    while newErr < oldErr * params['tolRatio'] or \
        (newErr > oldErr and resurrectedTrueFnalFlag):  # ie most recent change was 
    # non-trivial. The first condition is expected after a well-behaved iteration. The
    # second condition happens if an absent true fnal gets resurrected.
        previousNCR = nCR.copy()  # In case we overshoot, and want to revert
        if params['verbose']:
            print('newErr = ' + str(np.round(newErr, 2)))
        oldErr = newErr
        # Find an in-span fit equation and update the coefficients nC with it. Note that although
        # argin0 is the full list of coefficients, we only substitute among true functionals 
        # because argin2 and argin3 only contain linear fits among true functionals.
        nCR = updateLibCoeffsViaInSpanLoo_fn(nC, tC, rSqLooV, betasLooV, params['functionList'], 
                                             params['rSqThresh'], params['stepSize'], 
                                             params['verbose'])
          
        # Update nC etc:
        newErr = max(np.abs((nCR - tCR) / tCR))
        resurrectedTrueFnalFlag = np.sum(nCR != 0) > numActiveFnals
        numActiveFnals = np.sum(nCR != 0) 
        if newErr > oldErr and not resurrectedTrueFnalFlag: # case: undo this iteration
            if params['verbose']:
                print('newErr = ' + str(np.round(newErr, 2)) + ', undoing this iteration.')
            nCR = previousNCR
            newErr = oldErr 
        nC[tF] = nCR
    # End of while loop
    
    # Coefficient errors on true library functionals
    coeffErrR = np.abs((nCR - tCR) / tCR) 
            
    return nC, coeffErrR  

# End of findClosestLinearlyEquivalentVersion_fn 
#---------------------------------------------------------------------
 
def evolveModelAndCalcFoms_fn(coeffArray, recipes, initCond, timepoints, x, fftPow, 
                                        varLocMin, varLocMax, phase):
    """ For parallel processing."""
    
    print('starting evolution...\n')
    xTrainEvolved = evolveModel_fn(coeffArray, recipes, initCond, timepoints)
    print('done with evolution.\n')
    # xTrain is used for xDot predictions.
    # Calculate and print the FoMs. We only use the histograms, to compare the evolutions:
    fomDict = calculateFiguresOfMerit_fn(xTrainEvolved, x, fftPow, varLocMin, varLocMax, 
                                         timepoints, phase)
     
    return fomDict['histograms']
    #  np.array([[99,99,99],[99,99,99],[99,99,99]]) # 

# End of evolveModelAndCalcFoms_fn
#-------------------------------------------------------------------

def appendToLocalHistograms_fn(localHistograms, result):
    """ Helper to give a usable parallel processing result"""
    localHistograms.append(result)
 
# End of appendToLocalHistograms_fn
#-----------------------------------------------------------------------

def printModel_fn(coeffArray, variableNames, functionList, precision=2):
    """ Return a printable string version of a linear model.    

    Parameters
    ----------
    coeffArray : np.array of floats, numVars x numFunctionals 
    variableNames : list of str
    functionList : list of str
    precision : int, optional, default = 5.

    Returns
    -------
    mS : str (multi-line string version of model)

    """
    numFns = len(functionList)
    mS = ''
    for v in range(len(variableNames)):
        mS = mS + variableNames[v] + "' = "
        for j in range(numFns):
            if np.abs(coeffArray[v, j]) > 1e-5:
                signTag = ' + '
                if coeffArray[v, j] < 0:
                    signTag = ' - '
                mS = mS + signTag + str(np.round(np.abs(coeffArray[v, j]), precision)) + \
                    ' ' +  functionList[j] + ' '
        mS = mS + '\n'
    
    return mS
# End of printModel_fn
#-------------------------------------------------------------------
           
def updateLibCoeffsViaInSpanLoo_fn(nC, tC, rSqLooRow, betasLooRow, functionList, rSqThresh=0.95, 
                             stepSize=0.25, verbose=False):
    """
    An inner while loop, for one variable: given (i) a set of current coeffs of true library
    functionals; (ii) info about the true leave-one-out betas; and (iii) misc parameters, find a 
    functional with high Rsquared and also high error, and use its betas to make an incremental 
    substition into the coefficients equation. Go through the functionals via: first choose the 
    one with highest error. If its Rsquared from leave-one-out fit is high, use its fit equation
    eg x1 = b2*x2 + b3*x3 -> 0 = -x1 + b2*x2 + b3*x3, to modify the coefficients. If the Rsquared
    is too low, go to the functional with the next-highest error.
    
    Parameters
    ----------
    nC : np.array of floats, vector with len = numFnals. The current coefficients
    tC : ditto. The true coefficients 
    rSqLooRow : np.array of floats, vector with len = numFnals.
    betasLooRow : list (len = numFnals) of lists. Each inner list contains floats (len = number of 
               true library functionals)
    functionList : list of str, len = numFnals
    rSqThresh : float, default 0.95
    stepSize : float, default 0.25
    verbose : bool
    
    Returns
    -------
    nCR : np.array of floats, vector with length = number of true library functionals
    
    """
    indsRTried = []
    indsFTried = []
    tF = tC.astype(bool)
    nCR = nC[tF]  # the 'R' means Restricted, to just true fnals
    tCR = tC[tF]  # ditto 
    tC2 = tC.copy()
    tC2[tC2 == 0] = 1
    while len(indsRTried) < len(nCR):
        coeffErrR = np.abs((nCR - tCR) / tCR)  # Restricted to just true fnals
        coeffErrF = np.abs((nC - tC2 * tF) / tC2)  # 'F' means Full, ie over all fnals.
        # Remove already-tried inds from consideration:
        coeffErrR[indsRTried] = 0
        coeffErrF[indsFTried] = 0
        newErr = max(np.abs(coeffErrR))
        indR = np.where(coeffErrR == newErr)[0][0]  # ind with biggest error
        indF = np.where(coeffErrF == newErr)[0][0]  # we need this to get betasLoo
        # See if the loo fit has high enough rSq:
        if rSqLooRow[indF] > rSqThresh:
            if verbose:
                print('Using loo fit for ' + functionList[indF])
            b = betasLooRow[indF]  # the betas for fitting the left-out fnal using other 
            # fnals.
            # Let b equal "x1 = b2*x2 + b3*x3". Then we'll add some factor times the eqn
            # 0 = -1*x1 + b2*x2 + b3*x3 to the coeffs nCR
            factor = stepSize * (nCR[indR] - tCR[indR])  
            remainingFnalsCounter = 0
            for i in range(len(nCR)):
                if i == indR:
                    nCR[indR] = nCR[indR] + factor * (-1) 
                else:
                    nCR[i] = nCR[i] + factor * b[remainingFnalsCounter]
                    remainingFnalsCounter += 1 
            indsRTried = np.arange(len(nCR)) # to exit the inner while loop
        else:
            if verbose:
                print('True fnal ' + functionList[indF] + \
                      ' had highest error, but its Loo Rsq is too low  ' + \
                          '(' + str(np.round(rSqLooRow[indF], 2)) + ')')
            indsRTried.append(indR)
            indsFTried.append(indF) 
        
    return nCR 
 
# End of updateLibCoeffsViaInSpanLoo_fn
#---------------------------------------------------

def replaceNonLibraryFunctionals_fn(dC, tF, rSqRow, betasRow, functionList, rSqThresh, 
                                    verbose=False):
    """
    Given a set of coefficients and functionals, replace functionals that are not in the 'true'
    library using the in-span fit equation. Only do this if the Rsquared of the fit was high
    enough.

    Parameters
    ----------
    dC : np.array of floats, vector len = numFnals. Discovered coeffs for the variable at hand 
    tF : np.array of bools, vector len = numFnals. The true library functionals for the variable  
    rSqRow : np.array of floats, vector with len = numFnals.
    betasRow : list (len = numFnals) of lists. Each inner list contains floats (len = number of 
               true library functionals)
    functionList : list of str, len = numFnals
    rSqThresh : scalar float
    verbose : bool, default = False

    Returns
    -------
    nCLocal : np.array of floats, vector len = numFnals.

    """
    dF = dC.astype(bool)
    nCLocal = dC.copy()  # To store new (transformed coefficients)
    counter = 0
    for i in range(len(dF)):
        if dF[i] and not tF[i]:
            counter += 1
            if rSqRow[i] > rSqThresh:
                if verbose:
                    print('Replacing untrue fnal ' + functionList[i] + ' (Rsq = ' + \
                           str(np.round(rSqRow[i], 2)) + ')')
                b = betasRow[i]  # betasRow[i] are the coeffs in {fnal i = lin combo of true 
                # fnals}
                nCLocal[tF] = nCLocal[tF] + nCLocal[i] * b  # the [tF] ensures the correct cols get
                # updated.
                nCLocal[i] = 0  # since we subtracted out this term
            else:
                if verbose:
                    print('Untrue fnal ' + functionList[i] + ' was in the discovered model ' + \
                          'but is not in-span of true fnals (Rsq = ' + \
                          str(np.round(rSqRow[i], 2)) + ')')
    if counter == 0:
        if verbose:
            print('No untrue fnals for this variable')
    
    return nCLocal

# End of replaceNonLibraryFunctionals_fn
