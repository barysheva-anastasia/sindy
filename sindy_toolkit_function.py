import sys 
import numpy as np
from scipy.integrate import odeint
from numpy.linalg import lstsq  # since this accepts complex inputs and targets.
from multiprocessing import Pool
import time
import pickle

import warnings
warnings.filterwarnings("ignore", 
                        message='Casting complex values to real discards the imaginary part')
# To ignore console output from using only real part of FFTs. 

from toolkitSupportFunctions import generatePolynomialLibrary_fn, calculateLibraryFunctionalValues_fn, makeHammingWindow_fn, \
calculateSlopeAndStd_fn, calcWeightArray_fn, \
cullAndAssessWhetherToRerun_fn, smoothData_fn, parseTimepointsByMagnitudesOfVariables_fn, \
calculateDerivativesFromData_fn, calculateDerivativesFromModel_fn, estimateDerivatives_fn, \
combineSegmentCoeffs_fn, \
printWeightedCoeffModel_fn, evolveModel_fn, generateVariableEnvelopes_fn, \
calculateFiguresOfMerit_fn, findSpansOfFunctionalsLeaveOneOut_fn, \
printModel_fn, calculateFftForRegression_fn, weightedLeastSquares_fn, \
stabilizeLinearModel_fn, boundednessViolation_fn, trappingLinearModel_fn, penaltyLinearModel_fn

from plotFiguresOfMeritMosaics import plotFiguresOfMeritMosaics_fn
  
#%% Define any constraints on the initial library, in terms of functionals to ignore or functionals
# to permanently protect from culling.
def sindy_toolkit_func(train_data
                        , test_data
                        , datasetName
                        , variableNames
                        , polynomialLibraryDegree=2
                        , dt=1.
                        , marginInSecs = 10
                        , useFullLibraryFlag = True
                        , initialFunctionsToUseArray = np.array([])
                        , minNumStartIndsToUse = 100
                        , hammingWindowLengthForData = 20  # Hamming window for smoothing the data.
                        # Better too small than too big. Must satisfy: this / 2 <= margin, ie
                        # <= 2 * marginInSecs / dt, else smoothing reaches past the margin. #!!!!!#
                        , hammingWindowLengthForDerivsAndFnals = 10  # Hamming window for smoothing
                        # the derivatives and the library functionals. #!!!!!#
                        , maxFnValRatio = 50  # Timepoints are rejected where the active functionals
                        # differ in magnitude by more than this ratio. Used only if
                        # 'parseTimepointsByVarValuesFlag'. #!!!!!#
                        , regressOnFftAlpha = 0  # Fraction of the regression target that is the FFT
                        # rather than the raw derivative. 0 -> regress on the derivative only.
                        , balancedCullNumber = 20  # Prevents any one variable being culled down to
                        # nothing while other variables retain large libraries. Set = 100 (ie big)
                        # to disable. #!!!!!#
                        , numRegressionSegments = 6  # Number of separate regressions, on different
                        # collections of timepoints, whose coeffs are then combined via a
                        # median-type method. Must be > 'minNumSegmentResultsToUse' to leave
                        # any outliers to remove.
                        , minNumSegmentResultsToUse = 5  # We remove outlier coeffs down to this
                        # number, then take a median. Must be < 'numRegressionSegments'. #!!!!!#
                        , minAllowedWeightedCoeff = 0.001  # if a *weighted* coeff is smaller than
                        # this, cull the functional. #!!!!!#
                        , overlapFraction = 1  # Gives the number of points per segment, via:
                        # (1 + 2*overlapFraction) * (numTotalPoints / numRegressionSegments). Must
                        # satisfy (1 + 2*overlapFraction) < 'numRegressionSegments', else every
                        # segment uses all the timepoints and the segments are identical. #!!!!!#
                        , enforceBoundednessFlag = False  # Trapping-SINDy-style constrained
                        # optimization: after each regression and each cull, force the linear block
                        # A of the model to satisfy sym(A) <= 'boundednessMargin' * I, so that
                        # d/dt ||x||^2 <= 0 and the model provably produces bounded trajectories.
                        # Implemented for 'polynomialLibraryDegree' == 1 only; it is switched off
                        # with an 'Error:' message otherwise.
                        , boundednessMargin = 0.  # gamma, <= 0. 0 -> marginally bounded, which
                        # leaves the skew part of A alone and so preserves undamped oscillators
                        # (purely imaginary eigenvalues). Tightened automatically to
                        # 'strictBoundednessMargin' whenever the constant functional is active,
                        # since an affine term can grow along the kernel of sym(A). #!!!!!#
                        , strictBoundednessMargin = -1e-3  # The gamma used in place of
                        # 'boundednessMargin' when the latter is 0 but the constant functional is
                        # active. Trapping ball radius is then ||c|| / |gamma|. #!!!!!#
                        , boundednessRidgeLadder = (1., 10., 100., 1e3, 1e4)  # Escalating L2 pull
                        # toward the projected (bounded) matrix, refitting on the current support.
                        # The first rung that satisfies the constraint wins; if none do, a diagonal
                        # shift is applied, which satisfies it by construction. #!!!!!#
                        , boundednessMethod = 'ladder'  # How the constraint is enforced, once it
                        # is found to be violated. Both options return a model that satisfies it.
                        # 'ladder'   -> 'stabilizeLinearModel_fn'. Freeze the target at the
                        #               sym-part projection of the current A, walk
                        #               'boundednessRidgeLadder' upward refitting toward it row by
                        #               row, take the first feasible rung. Cheap.
                        # 'trapping' -> 'trappingLinearModel_fn'. The relax-and-split scheme of
                        #               Kaptanoglu et al. 2021 specialized to a linear library:
                        #               alternate a joint solve over all active entries with
                        #               re-projecting the auxiliary matrix from the CURRENT sym(A),
                        #               annealing 'boundednessNuLadder' downward until feasible.
                        #               Keeps the derivative fit far closer to the unconstrained
                        #               one (median R^2 +0.15 vs -12.8 on planted linear systems)
                        #               but costs ~46x more and did not improve trajectories there;
                        #               see 'benchmark_trapping_vs_ladder.py'. Falls back to the
                        #               ladder whenever it declines a problem as too large. #!!!!!#
                        # 'penalty'  -> 'penaltyLinearModel_fn'. Put the unboundedness measure
                        #               itself into the regression loss: minimize
                        #               ||Xb - y||^2_w + lam * max(0, max eig sym(A) - gamma),
                        #               smoothed, by accelerated gradient descent, escalating lam
                        #               along 'boundednessPenaltyLadder'. Unlike the other two the
                        #               quantity minimized IS the quantity tested; the penalty has
                        #               a bounded gradient, so it is exact and a finite lam lands
                        #               strictly inside the feasible set rather than on its
                        #               boundary. Falls back to the ladder when it declines a
                        #               problem as too large, exactly as 'trapping' does. #!!!!!#
                        , boundednessNuLadder = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
                        # Decreasing relaxation strengths for 'trapping', warm-started along the
                        # ladder. Must reach ~1e-6 for normType 'identity' to become feasible at
                        # all; 'spectral' is met several rungs earlier. #!!!!!#
                        , boundednessPenaltyLadder = (1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3)
                        # Escalating penalty weights for 'penalty', warm-started along the ladder,
                        # first feasible rung wins. These are MULTIPLES of an automatic scale,
                        # ||dataHessian||_2 * (violation - gamma), since lam trades a squared
                        # residual against an eigenvalue and so has no dataset-independent absolute
                        # value. The balance argument in 'penaltyLinearModel_fn' puts feasibility
                        # near a multiple of 1, so the ladder brackets that by three orders each
                        # way. #!!!!!#
                        , boundednessPenaltySmoothing = 0.25  # Smoothing of the penalty for
                        # 'penalty', as a fraction of the CURRENT violation gap, re-derived at each
                        # rung. Larger is better conditioned and more conservative. The accepted
                        # model ends up roughly this fraction of |gamma| inside the margin. #!!!!!#
                        , boundednessMaxNumFreeEntries = float('inf')  # 'trapping' and 'penalty'
                        # both build a dense system of this size, so they decline above this many
                        # active library entries and the ladder is used for that call instead.
                        # Default is now unbounded, so the chosen method always runs regardless of
                        # library size (no more silent fallback to the ladder on large libraries);
                        # pass a finite value to restore the old size cap. #!!!!!#
                        , boundednessNormType = 'spectral'  # How the Lyapunov weight P in the
                        # trapping condition A'P + PA <= 2*gamma*P is handled.
                        # 'identity' -> P fixed to I, ie literally the trapping-SINDy condition
                        # sym(A) <= gamma*I. 'spectral' -> P left free, which reduces the test to
                        # max Re eig(A) <= gamma. 'spectral' is the default because P = I is
                        # strongly conservative for non-normal systems, and the linear systems from
                        # 'core/matrix.py::generate_matrix' are exactly that: under 'identity' the
                        # constraint excludes the true model and the derivative-fit R^2 measured on
                        # them falls from ~0.34 to ~-31, where 'spectral' keeps it at ~0.30.
                        # See 'stabilizeLinearModel_fn'.
                        , maxNumIterations = None  # Hard cap on regress-cull iterations
                        # ('whichIter'), applied independently to EACH training trajectory (its
                        # counter starts at 0 every time). None -> no cap, ie the existing
                        # 'maxNumCulls'-based stopping rule is the only limit. Use to bound
                        # worst-case runtime on datasets with many variables/iterations.
                        , maxRunTimeSecs = None  # Hard wall-clock budget (seconds), applied
                        # independently to EACH training trajectory (its clock starts fresh at the
                        # start of that trajectory's regress-cull loop), checked once per iteration.
                        # None -> no time limit. So every trajectory gets the same budget
                        # regardless of how long earlier trajectories took; it is not a shared
                        # budget for the whole call.
                        ):
    # Library:

    permanentlyProtectedFunctions = np.array([])

    #---------------------

    seed = np.random.randint(0, high = 1e4)
    np.random.seed(seed)

    outputFilenamePrefix = 'outputFile'  # We'll append system name, initial or second run, and seed
    pickleFilenamePrefix = 'runResults'  # We'll append the above items plus .pickle

    #--------------------------------------

    #%% System details:

    numTrajTrain = train_data.shape[0]  # number of training trajectories (also act as validation)
    numTrajTest = test_data.shape[0]  # number of test trajectories


    numSecsInTrain = train_data.shape[1] * dt


    #------------------------------------------

    # Plotting parameters: 
    showFiguresOfMeritMosaicsFlag = True 
    # got sparser via culling.
    showTimeSeriesPlotsAtEndOfRunFlag = False


    #%%  Smoothing parameters:
    # 'hammingWindowLengthForData' and 'hammingWindowLengthForDerivsAndFnals' are now argins.
    # Better too small than too big. #!!!!!#

    # marginInSecs = 10 # the number of seconds to throw out at the start and end. A convenience to 
    # prevent endpoint artifacts in the smoothing windows, so 0.5 * window length (in secs) suffices.
    
    #%% Parameters for linear regression

    # 'regressOnFftAlpha' is now an argin.
    fftRegressionTarget = 'magnitude'

    parseTimepointsByVarValuesFlag = True  # Says whether to use the 'maxFnValRatio' parameter.
    # 'maxFnValRatio' is now an argin.

    printDiagnosticsForParsingFlag = False
    printWeightedCoefficientModelFlag =  True 

    # Weight timepoints for the regressions:
    weightTimepointsFlag = True
    scaleByLogFlag = True 

    # Parameters related to doing several regressions, over different collections of timepoints, and 
    # deciding the final coeffs via a median-type method:
    # 'numRegressionSegments' is now an argin.
    # 'overlapFraction' is now an argin. It gives the number of points per segment, via:
    # (1 + 2*overlapFraction)*(numTotalPoints / numRegressionSegments).

    # 'minNumSegmentResultsToUse' is now an argin. We remove outlier coeffs down to this number,
    # then take a median.

    snrThreshold = 0.4  # Used to remove segments when combining coefficents from the segment 
    # regressions. Not a sensitive parameter if not too small.

    # Diagnostic to show spread of coeffs over various segments and to see how segment coeff estimates
    # are combining: (To disable: -1 -> print nothing)
    printCoeffsForEachSegmentThresh = -1 # 9  # Diagnostic. If this var has fewer library fns than this
    # threshold, print the coeffs over each of its fns from each segment, eg to see spread of coeff
    # estimates.


    numDtStepsForDeriv = 2   # how many dt's to use for euler or 4th order stepping.
    numEvolveStepsForEvolutionTarget = 50  # If regressionTarget = 'evolution', how many steps to
    # 'evolve' by stringingtogether 4th order derivatives at a sequence of points.


    #%% Culling step parameters:

    extraNumFnsCulledPerIter = 1
    # 'minAllowedWeightedCoeff' is now an argin. #!!!!!# If a *weighted* coeff is smaller than it,
    # cull the functional.

    almostZeroRoundDownThreshold = 1e-5

    # 'balancedCullNumber' is now an argin. #!!!!!# It prevents any one variable being culled down
    # to nothing while other variables retain large libraries. Set = 100 (ie big) to disable.

    stopIfVarIsCulledFlag = True  # prevents wasted time, but it gets automatically set to False if 
    # there are extra noise variables, because we expect the system to first kill the noise variables 
    # then re-start on the remaining variables.

    cullUsingInSpanFlag = True  # As part of iteration, for each variable: test whether any
    # active functional is within the span of the remaining active functionals (leave one out). If yes,
    # cull it. Break ties by choosing the functional with the lowest weighted coefficient.
    inSpanCullThreshold = 0.99 #!!!!!#  # If Rsq value of fit to a functional is above this, can cull.
    # set > 1 to disable.
    percentileThresholdForInSpanCulling = 0.2  # weighted coeff must be below this percentile
    # (among functionals of that var) to be culled.
    printDiagnosticOfInSpanCullFlag = True  # True -> print Rsq values each iteration
    #----------------------------------------------------------
    # Weight the current coeffs during culling to account for large differences in values taken on by
    # the functions they multiply:

    # Define a percentile for use in setting functional weights. The goal is to compensate for the
    # larger number of functionals with high degree (resulting from combinatorics):
    percentileOfImputedValuesForWeights = 10  #!!!!!# a low number shifts preference towards 
    # lower-valued functionals, which may effectively mean lower-degree functionals.
    # The goal is to offset the combinatorially larger number of high-degree functionals (eg using the
    # median would favor high-degree functionals because there are so many).
    coeffWeightsCutoffFactor = 10  #  Must be >= 1, where  1 -> No weighting of coeffs. This defines
    # the allowable differences in scale for typical functional values. A high value -> big contrast 
    # in weights for various functionals. Coeff weights will be clipped at [centralValue/this,
    # centralValue*this]
    # Note: not weighting the coeffs disadvantages high-degree functionals.

    #-------------------------------------------------------------
    #%% Figures of Merit (FoM):

    useTrueInitialValueForSimFlag = False  # If True, we start the final train set simulation from the
    # true, clean data values. If False, we use an estimate (realistic use-case situation). 

    windowLengthForFomEnvelopes = 30  # Length of window for max-min envelopes
    shrinkFomEnvelopeFactor = 0.1 #!!!!!#
    # If the data is very noisy, 'inEnvelopeFoM' means nothing, so shrink the envelope. If the data
    # is not as all noisy, increase the envelope size. The goal is to make the 'inEnv' fom meaningful.

    # For fft power:
    numFftPoints = 50

    numEvolutionsForTrainFom = 1  #!!!!!# To check for stability. Downside: unstable evolutions can take a
    # long time to run. So ideally add bail-out option to odeint
    numEvolutionsForValFom = numEvolutionsForTrainFom


    doNotEvolveAfterInSpanCullFlag = True  # To save time, since in-span culling mostly happens in
    # very early stages.
    maxNumFunctionalsToEvolve = 100  # Because with very many functionals, solve_ivp can hang,
    # ruining the run.  A high number #!!!!!#effectively disables this.

    #--------------------------------------------------------------------
    # Parameters to restore and protect culled functionals:
    restoreBasedOnFoMsFlag = True #!!!!!#
    startAfterWhichIter = 8  # do not restore until part-way in
    numItersProtection = 3 # Once restored, protect from culling for this many iterations.

    # The FoMs to monitor, and a vector that contains (0) their triggering value drops (expressed as a
    # fraction of previous iteration's value); and (1) the minimum value for the previous iteration
    # (the point is to only restore functionals if the previous result was decent).
    # This is only relevant if 'restoreBasedOnFoMsFlag' = True
    fomChangesDict = {'inBounds':[0.8, 0.8], 'evolutionsCorrelation':[0.8, 0.98],
                    'histogramCorrelation':[0.8, 0.6], 'inEnvelopeFoM':[0.8, 0.6],
                    'fftPower':[0.8, 0.7],'stdDevFoM':[0.3, 0.8]}  #!!!!!# For restoring purposes,
    # stdDevFoM will be subtracted from one, so we look for values close to 0 changing to values far
    # from 0. Note that the first arg for stdDevFoM is a threshold on Subtraction, not Ratio (the
    # other change detectors use Ratio).
    # NOTE: 'results', defined during the culling phase, must have keys that match the keys in
    # 'fomChangesDict'. If they do not, an error message will print to console, but still manually 
    # comparing 'results' and 'fomChangesDict' before running is best.

    #--------------------------------------------------

    # Initialize lists to save FoMs from each iteration. This is necessary to do here, so we can then 
    # define 'results', the histories relevant to restoring functionals.
    historyCoeffArray = []  # to save coeff matrices from each step.
    historyProtectedFnsArray = []
    historyWhichIter = []
    historyWhichCull = []
    historyInBoundsFoM = []
    historyInEnvelopeFoM = []
    historyStdDevFoM = []
    historyMeanFoM = []
    historyMedianFoM = []
    historyFftCorrelationFoM = []
    historyFftPower = []
    historyHistogramCorrelationFoM = []
    historyHistograms = []
    historyHistogramBins = []
    historyMinHistCorrelationForEvolutions = []
    historyXDotDiffEnvelope = []
    historyXDotInBoundsFoM = []
    historyXDotHistogramCorrelationFoM = []
    historyXDotHistograms = []
    historyXDotHistogramBins = []

    # for xVal:
    historyCoeffArrayVal= []  # to save coeff matrices from each step.
    historyProtectedFnsArrayVal = []
    historyWhichIterVal = []
    historyWhichCullVal = []
    historyInBoundsFoMVal = []
    historyInEnvelopeFoMVal= []
    historyStdDevFoMVal = []
    historyMeanFoMVal = []
    historyMedianFoMVal = []
    historyFftCorrelationFoMVal = []
    historyFftPowerVal = []
    historyHistogramCorrelationFoMVal = []
    historyHistogramsVal = []
    historyHistogramBinsVal = []
    historyMinHistCorrelationForEvolutionsVal = []
    historyXDotDiffEnvelopeVal = []
    historyXDotInBoundsFoMVal = []
    historyXDotHistogramCorrelationFoMVal = []
    historyXDotHistogramsVal = []
    historyXDotHistogramBinsVal = []



    cullingRulesDict = {'setConstantDerivEstimatesToZero':True,
                        'extraNumFnsCulledPerIter':extraNumFnsCulledPerIter,
                        'minAllowedWeightedCoeff':minAllowedWeightedCoeff}

 

    """ ------------------------- END USER ENTRIES --------------------------------------------- """

    #%%

    """ ------------------------------ BEGIN MAIN ---------------------------------------------- """

    """ One-time items: """

    initialRunTag = '_initialRun_'
    if not useFullLibraryFlag:
        initialRunTag = '_secondRun_'
    outputFilename = outputFilenamePrefix + '_' + datasetName + initialRunTag + str(seed)
    pickleFilename = pickleFilenamePrefix + '_' + datasetName + initialRunTag + str(seed) + '.pickle'
    
    print('')
    print('See ' + outputFilename + ' for detailed progress of run.')
    print('Load ' + pickleFilename + ' to get complete run data after completion. \n ')

    if useFullLibraryFlag:
        initialFunctionsToUseArray = []

    # if results.keys() != fomChangesDict.keys():
    #     print('Caution: results keys and fomChangesDict keys must match. Fix and restart.')
    #     print('results.keys are: ', results.keys())
    #     print('fomChangesDict.keys are: ', fomChangesDict.keys())

    if fftRegressionTarget == 0:
        fftRegressionTarget = 'none'  # just for book-keeping

        
    culledFunctionIndices = []
    numCullsWithNoChange = 0
    newRemovedFunctionStr = '[]'

    #%% check that the padding + window will fit inside the margin:
    margin = int(np.round(marginInSecs / dt))    # margin = number of timepoints in the boundary
    # regions at start and at end.
    if hammingWindowLengthForData/2 > margin:
        print('Error: hammingWindowLength and/or padFactor are too large: ' + \
            str(np.round(hammingWindowLengthForData/2)) + ' > ' + str(np.round(margin)))

    #%% check that the regression segment parameters are mutually consistent:
    # Each segment draws (1 + 2*overlapFraction) / numRegressionSegments of the available timepoints,
    # so if that ratio reaches 1 every segment uses all of them and the segments are identical. The
    # number of points drawn is clamped below, so this only degrades the results, it does not fail.
    if 1 + 2 * overlapFraction >= numRegressionSegments:
        print('Error: overlapFraction is too large for numRegressionSegments: ' + \
            str(1 + 2 * overlapFraction) + ' >= ' + str(numRegressionSegments) + \
            '. Every segment will use all the timepoints, so the segments will be identical and ' + \
            'combining their coeffs will do nothing.')
    # 'minNumSegmentResultsToUse' is a floor on the outlier-removal loop in combineSegmentCoeffs_fn,
    # so it must leave at least one coeff to remove:
    if numRegressionSegments <= minNumSegmentResultsToUse:
        print('Error: numRegressionSegments is too small for minNumSegmentResultsToUse: ' + \
            str(numRegressionSegments) + ' <= ' + str(minNumSegmentResultsToUse) + \
            '. Outlier removal will have no coeffs to trim.')

    #%% check that the boundedness constraint is applicable. The trapping condition reduces to
    # sym(A) <= gamma*I only when the model has no quadratic (or higher) terms; above degree 1 it
    # would need the full search for a trapping-region center, which is not implemented here:
    if enforceBoundednessFlag and polynomialLibraryDegree != 1:
        print('Error: enforceBoundednessFlag is implemented for polynomialLibraryDegree == 1 ' + \
            'only, but polynomialLibraryDegree = ' + str(polynomialLibraryDegree) + \
            '. Disabling the boundedness constraint.')
        enforceBoundednessFlag = False
    if enforceBoundednessFlag and boundednessMargin > 0:
        print('Error: boundednessMargin must be <= 0, but is ' + str(boundednessMargin) + \
            '. Using 0 instead.')
        boundednessMargin = 0.
    if enforceBoundednessFlag and boundednessNormType not in ('spectral', 'identity'):
        print('Error: boundednessNormType must be "spectral" or "identity", but is ' + \
            str(boundednessNormType) + '. Using "spectral".')
        boundednessNormType = 'spectral'
    if enforceBoundednessFlag and boundednessMethod not in ('ladder', 'trapping', 'penalty'):
        print('Error: boundednessMethod must be "ladder", "trapping" or "penalty", but is ' + \
            str(boundednessMethod) + '. Using "ladder".')
        boundednessMethod = 'ladder'

    def enforceBoundedness_fn(coeffArray, functionsToUseArray, designMatrices):
        """
        Apply the boundedness constraint by whichever method 'boundednessMethod' selects.

        Wrapped so that the two call sites below stay identical, and so that 'trapping' and
        'penalty' can hand oversized problems back to the ladder rather than returning an
        unconstrained model: the toolkit integrates every candidate to score figures of merit, so a
        call that silently does nothing would break the invariant the whole flag exists to provide.
        """
        if boundednessMethod in ('trapping', 'penalty'):
            if boundednessMethod == 'trapping':
                newCoeffArray, newLib, action, numEigCalls = trappingLinearModel_fn(
                    coeffArray, functionsToUseArray, designMatrices, len(variableNames),
                    boundednessMargin, strictBoundednessMargin, boundednessNuLadder, variableNames,
                    outputFilename, boundednessNormType,
                    maxNumFreeEntries=boundednessMaxNumFreeEntries)
            else:
                newCoeffArray, newLib, action, numEigCalls = penaltyLinearModel_fn(
                    coeffArray, functionsToUseArray, designMatrices, len(variableNames),
                    boundednessMargin, strictBoundednessMargin, boundednessPenaltyLadder,
                    variableNames, outputFilename, boundednessNormType,
                    maxNumFreeEntries=boundednessMaxNumFreeEntries,
                    smoothingFactor=boundednessPenaltySmoothing)
            if action != 'skipped-too-large':
                return newCoeffArray, newLib, action, numEigCalls

            newCoeffArray, newLib, action, ne = stabilizeLinearModel_fn(
                coeffArray, functionsToUseArray, designMatrices, len(variableNames),
                boundednessMargin, strictBoundednessMargin, boundednessRidgeLadder, variableNames,
                outputFilename, boundednessNormType)

            return newCoeffArray, newLib, 'ladderFallback:' + action, numEigCalls + ne

        return stabilizeLinearModel_fn(
            coeffArray, functionsToUseArray, designMatrices, len(variableNames), boundednessMargin,
            strictBoundednessMargin, boundednessRidgeLadder, variableNames, outputFilename,
            boundednessNormType)

    numTimepointsForFoM = int((numSecsInTrain - 2*marginInSecs) / dt)


    #%% Define train and test trajectories:

    # Note re test trajectories: Since these are used as a holdout set, so we do less processing
    # than for xTrain. We really just need enough to calculate a model's FoMs on the test set.
    # Test trajectories get the following processing: temporarily add noise, solely to calculate an
    # envelope for the 'inEnvelope' FoM; collect stats like std dev, for use in FoMs; calculate
    # xDotTest for the xDot FoMs. All tTest timepoints are used for FoMs on test trajectories.

    tTrain = np.arange(0, numSecsInTrain, dt)
    numSecsInTest = numSecsInTrain
    tTest = np.arange(0, numSecsInTest, dt)

    numVars = train_data.shape[2]

    xTrainAll = train_data
    xTestAll = test_data


    # The following is for plotting only.
    xDotTrainOfNoisyDataAll = []
    for j in range(numTrajTrain):
        xDotTrainOfNoisyDataAll.append(calculateDerivativesFromData_fn(xTrainAll[j], tTrain))  # from
        # the noisy, unsmoothed data.

        
    # Preparations

    #%% Generate Hamming filter windows:
    # For use on Data; also calculate some windowing values:
    hammData = makeHammingWindow_fn(hammingWindowLengthForData)
    halfData = int(np.floor(hammingWindowLengthForData) / 2)

    # For use on Derivatives:
    hammDerivs = makeHammingWindow_fn(hammingWindowLengthForDerivsAndFnals)

    # For use on library functions:
    hammForFunctions = makeHammingWindow_fn(hammingWindowLengthForDerivsAndFnals)

    #%% Calculate global and local max and min for each variable, used to calculate FoMs of the models
    # at each iteration:
    fomTimepointInds = np.array(range(margin, margin + numTimepointsForFoM))
    maxPhaseShift = int(0.1 / dt)  # to reduce FoM error for otherwise-correct evolutions that are out
    # of phase.
    # For each trajectory, get local measures of max and min, per variable:
    varLocalMaxTrainAll = []
    varLocalMinTrainAll = []
    for j in range(numTrajTrain):
        this = xTrainAll[j]
        varLocalMax, varLocalMin = \
            generateVariableEnvelopes_fn(this[fomTimepointInds, :], windowLengthForFomEnvelopes, 
                                        dt, shrinkFomEnvelopeFactor)
        varLocalMaxTrainAll.append(varLocalMax)
        varLocalMinTrainAll.append(varLocalMin)

    varLocalMaxTestAll = []
    varLocalMinTestAll = []
    for j in range(numTrajTest):
        this = xTestAll[j]
        varLocalMaxTest, varLocalMinTest = \
            generateVariableEnvelopes_fn(this, windowLengthForFomEnvelopes, dt, 
                                        shrinkFomEnvelopeFactor)
        varLocalMaxTestAll.append(varLocalMax)
        varLocalMinTestAll.append(varLocalMin)

    #%% Calculate timepoint weights to use in linear regression.
    # Do this for each variable, for each time-point. Procedure: Use the noisy data. For each Point,
    # (a) fit a line to the points in a window around the Point; (b) tilt the points so that the
    # fitted line has slope 0; (c) define a distribution using the points in the window; (d) the
    # weight is 1 / z-score of the Point relative to the distribution.
    # Also, within the loop maybe clip or shrink the raw initial data to save flops. This combination
    # of actions makes this section a bit messy (and clip and shrink don't work anyway).

    print('calculating weights and smoothing.')
    pointWeightsAll = []
    smoothedXTrainAll = []
    eulerSlopesAll = []  # In case we chose to use this as a derivative estimate. Could probably be
    # refactored out.
    pointWeightsForFomAll = []

    for j in range(numTrajTrain):
        thisXTrain = xTrainAll[j].copy()
        thisPointWeights =  np.ones(thisXTrain.shape)
        thisSmoothedXTrain= np.zeros(thisXTrain.shape)
        thisEulerSlopes = np.zeros(thisXTrain.shape)
        for i in range(numVars):  # loop over variables.
            temp = thisXTrain[:, i]
            slopeX, stdX, meanX = \
                calculateSlopeAndStd_fn(temp, dt, hammingWindowLengthForDerivsAndFnals)
            # argouts 2 and 3 are used for weights.

            # A. Update pointWeights array:
            if weightTimepointsFlag:
                weightRow = np.ones(temp.shape)
                mask = np.abs(temp - meanX) > 1e-9
                weightRow[mask] = stdX[mask] / np.abs(temp[mask] - meanX[mask])  # 1 / z-score.
                # Timepoints with super low z-scores remain at maxWeight.
                # Modify to basic weights if wished: 
                if scaleByLogFlag:
                    weightRow =  np.log(weightRow + 1)  # base e, could be another base
                # Insert this row into weights array:
                thisPointWeights[:, i] = weightRow

            # B. Update eulerSlopes (for possible future use as derivative estimates):
            thisEulerSlopes[:, i] = slopeX
    

        # Also smooth with spline or hamming if wished:
        # 3. Hamming is done here to keep the logical flow clear: 
        thisXTrain = smoothData_fn(thisXTrain, hammData)

        xTrainAll[j] = thisXTrain  # replace original noisy xTrain with smoothed version.
        pointWeightsAll.append(thisPointWeights)
        smoothedXTrainAll.append(thisXTrain)

        # Define point weights for FoM points, summing to 1:
        thisPointWeightsForFom = thisPointWeights[fomTimepointInds, :].copy()
        pointWeightsForFomAll.append(thisPointWeightsForFom / \
                                np.sum(thisPointWeightsForFom, axis=0))  # broadcast, not tiled

    # Note that smoothing updates xTrain itself.

    # в душе не ебу что происходит в этом параграфе
    #%% Using point weights, select starting points for solve_ivp evolutions: use the highest weight
    # points close to 'margin', which is the point at the end of a short opening segment that we
    # ignore (to avoid transient effects). The goal is to legally maximize the chance that the
    # evolution will start accurately. This is maybe a frivolous detail, and we could just use
    # simInitConds = xTrain[margin,:]
    simInitCondsAll = []
    for j in range(numTrajTrain):
        thisXTrain = xTrainAll[j]
        thisPointWeights = pointWeightsAll[j]
        if useTrueInitialValueForSimFlag:
            simInitCondsAll.append(xTrain[j][margin, :].copy())
        else:
            temp1 = margin * np.ones(thisXTrain[margin, :].shape)
            for i in range(numVars):
                temp2 = thisPointWeights[margin - 2:margin + 3, i]
                ind = np.where(temp2 == max(temp2))[0][0] + margin - 2  # the 2nd [0] in case of a tie.
                temp1[i] = thisXTrain[ind, i]
            simInitCondsAll.append(temp1)

    #%% Now that raw data is smoothed, calculate the Derivatives using the pre-processed data, and
    # maybe smooth them. Also calculate some stats needed for FoMs on xDot.
    print('Smoothing initial derivatives')
    xDotTrainAll = []
    xDotTrainUnsmoothedAll = []
    for j in range(numTrajTrain):
        thisXDot = calculateDerivativesFromData_fn(xTrainAll[j], tTrain)
        thisXDot = smoothData_fn(thisXDot, hammDerivs)
        xDotTrainAll.append(thisXDot)
        xDotTrainUnsmoothedAll.append(xDotTrainAll[j].copy())  # For possible plotting later.

    # For FoMs, calculate local max and mins (FFT power spectrum is calculated later):
    xDotLocalMaxAll = []
    xDotLocalMinAll = []
    for j in range(numTrajTrain):
        thisXDotTrain = xDotTrainAll[j]
        temp1, temp2 = generateVariableEnvelopes_fn(thisXDotTrain[fomTimepointInds, :],
                                                    windowLengthForFomEnvelopes, dt, 
                                                    shrinkFomEnvelopeFactor)
        xDotLocalMaxAll.append(temp1)
        xDotLocalMinAll.append(temp2)

    # xDotLocalMaxTestAll = []
    # xDotLocalMinTestAll = []
    # for j in range(numTrajTest):
    #     thisXDotTest = xDotTestTrueAll[j]
    #     temp1, temp2 = generateVariableEnvelopes_fn(thisXDotTest, windowLengthForFomEnvelopes, dt,
    #                                                 shrinkFomEnvelopeFactor)
    #     xDotLocalMaxTestAll.append(temp1)
    #     xDotLocalMinTestAll.append(temp2)

    #%% Create an initial library of functionals. Also do some prep work with it (4 steps):

    # Step 1. Generate the library:
    functionList, recipes = generatePolynomialLibrary_fn(variableNames, polynomialLibraryDegree)

    # Get the index of the non-constant functions for later use:
    constantFnIndex = np.where(np.array(functionList) == '1')[0]
    temp = np.array(range(len(functionList)))
    nonConstantIndices = temp[temp != constantFnIndex]  # To mark non-constant functionals.

    # Initialize boolean arrays to use as modelActiveLib, one for each train trajectory. Ditto
    # for float arrays for modelCoefs. These will be useful when selecting a group of
    # models (one for each training trajectory) and plotting their evolutions of all trajectories.
    functionsToUseArrayAll = []

    # Use a full all-ones array, or use a custom initial functions array:
    if useFullLibraryFlag: # test for emptiness
        initialFunctionsToUseArray = np.ones((len(variableNames), len(functionList)), dtype = bool)
    for j in range(numTrajTrain):
        functionsToUseArrayAll.append(initialFunctionsToUseArray)

    coeffArrayAll = []
    for j in range(numTrajTrain):
        coeffArrayAll.append(np.zeros((len(variableNames), len(functionList))))

    #%% Step 2. Calculate values of the feature library, for use during regressions:
    LAll = []
    LNoisyAll = []
    for j in range(numTrajTrain):  # the library functions, calculated at each time point.
        # 'L' is numSamples x numFunctions. Each row is the value of all functions at one timepoint. Each
        # column is the value of one function at all timepoints.
        LAll.append(calculateLibraryFunctionalValues_fn(xTrainAll[j], recipes))
        LNoisyAll.append(LAll[j].copy())  # Same functional values as LAll (both are evaluated on
        # the already-smoothed xTrain), so reuse them. LAll gets smoothed again below, hence a copy.

    LTestAll = []
    for j in range(numTrajTest):
        LTestAll.append(calculateLibraryFunctionalValues_fn(xTestAll[j], recipes))

    # Smooth the library functions:
    print('smoothing library of functionals...')
    for j in range(numTrajTrain):
        LAll[j] = smoothData_fn(LAll[j], hammForFunctions)  # 'L' stands
    # for 'library', as in 'the values at each timepoint of all the functionals in the library'.

    #%% Step 3. Make weights for coeffs, for use in culling functions. Base this on mean and std of
    # xTrain values. The weightVector is big if the functional values can get big, and it's small if
    # the functional values tend to stay small.
    imputedSizeOfFunctionalsAll = []# 'imputedSizeOfFunctionals' is a positive, mu + sigma estimate of
    # functional magnitude. It's used at each regression to estimate a weightVector for the relevant
    # functionals.
    if coeffWeightsCutoffFactor > 1:
        skip = 5  # To save compute time.
        for j in range(numTrajTrain):
            functionalVals = LAll[j][margin:-margin:skip, :]
            medianFunctionalVals = np.median(functionalVals, axis=0)
            imputedSizeOfFunctionalsAll.append(np.abs(medianFunctionalVals + \
                np.std(functionalVals, axis=0) * np.sign(medianFunctionalVals)))
    else:
        imputedSizeOfFunctionalsAll.append(np.ones((1, len(functionList))))
    
    # Print functional library:
    console = sys.stdout
    with open(outputFilename, 'a') as file:
        print('Function library: \n' + str(functionList) + '\n',  file=file)
        sys.stdout = console
        file.close()

    # We are done preparing the functional library.

    #%% Collect some stats about the post-smoothed time-series, for use with FoMs later:
    stdDevTrainAll = []
    meanTrainAll = []
    medianTrainAll = []
    xTrainFftPowerAll = []
    xTrainHistogramAll = []
    xTrainHistogramBinsAll = []

    for j in range(numTrajTrain):
        thisXTrain = xTrainAll[j]
        stdDevTrainAll.append(np.std(thisXTrain[fomTimepointInds, :], axis=0))
        meanTrainAll.append(np.mean(thisXTrain[fomTimepointInds, :], axis=0))
        medianTrainAll.append(np.median(thisXTrain[fomTimepointInds, :], axis=0))
        # FFT power spectrum:
        thisXTrainFftPower = np.zeros((numFftPoints, numVars))
        for i in range(numVars):
            x = thisXTrain[fomTimepointInds, i].copy()
            x = x - np.mean(x)
            xP = pow(np.real(np.fft.fft(x)), 2) # power spectrum of fft
            thisXTrainFftPower[:, i] = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])  # crop,
            # normalize
        xTrainFftPowerAll.append(thisXTrainFftPower)
        # histograms of time-series:
        numHistogramBins = 100
        thisXTrainHistogram = np.zeros((numHistogramBins, numVars))
        thisXTrainHistogramBins = np.zeros((numHistogramBins, numVars))
        for i in range(numVars):
            temp = np.histogram(thisXTrain[fomTimepointInds, i].flatten(), bins=numHistogramBins)
                # Range of histogram is automatically set to (min, max)
            thisXTrainHistogram[:, i] = temp[0]
            thisXTrainHistogramBins[:, i] = temp[1][0:-1]
        xTrainHistogramAll.append(thisXTrainHistogram)
        xTrainHistogramBinsAll.append(thisXTrainHistogramBins)

    # Ditto for xTest:
    stdDevTestAll = []
    meanTestAll = []
    medianTestAll = []
    xTestFftPowerAll = []
    xTestHistogramAll = []
    xTestHistogramBinsAll = []

    for j in range(numTrajTest):
        thisXTest = xTestAll[j]
        stdDevTestAll.append(np.std(thisXTest, axis=0))
        meanTestAll.append(np.mean(thisXTest, axis=0))
        medianTestAll.append(np.median(thisXTest, axis=0))
        # FFT power spectrum:
        thisXTestFftPower = np.zeros((numFftPoints, numVars))
        for i in range(numVars):
            x = thisXTest[:, i].copy()
            x = x - np.mean(x)
            xP = pow(np.real(np.fft.fft(x)), 2) # power spectrum of fft
            thisXTestFftPower[:, i] = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])  # crop,normalize
        xTestFftPowerAll.append(thisXTestFftPower)
        # histograms of time-series:
        numHistogramBins = 100
        thisXTestHistogram = np.zeros((numHistogramBins, numVars))
        thisXTestHistogramBins = np.zeros((numHistogramBins, numVars))
        for i in range(numVars):
            temp = np.histogram(thisXTest[:, i].flatten(), bins=numHistogramBins)  # range is
                # automatically set to (min, max)
            thisXTestHistogram[:, i] = temp[0]
            thisXTestHistogramBins[:, i] = temp[1][0:-1]
        xTestHistogramAll.append(thisXTestHistogram)
        xTestHistogramBinsAll.append(thisXTestHistogramBins)

    #%% Calculate calculate FFTs of xDotTrain and of the functional values L, for possible use in
    # regressions; This can be the full fft (real + imag), real-only, or power (determined by flag).
    # Also calculate FFT power spectrum of xDotTrain (not needed for L) for use in FoMs.

    # Training trajectories:
    fftXDotTrainTargetAll = []
    fftPowerXDotTrainAll = []
    fftLibFnsAll = []
    xDotTrainHistogramAll = []
    xDotTrainHistogramBinsAll = []
    for j in range(numTrajTrain):
        thisXDotTrain = xDotTrainAll[j]
        thisL = LAll[j]
        thisFftXDotTrainTarget = np.zeros((numFftPoints, numVars), dtype=complex)  # truncated FFT,
                                                                                # used for regression
        thisFftPowerXDotTrain = np.zeros((numFftPoints, numVars))  # FFT power spectrum, used for FoMs
        for i in range(numVars):
            # 1. Some form of FFT:
            thisX = thisXDotTrain[fomTimepointInds, i].copy()
            fftTarget = calculateFftForRegression_fn(thisX, numFftPoints, fftRegressionTarget)
            thisFftXDotTrainTarget[:, i] = fftTarget
            # 2. FFT power spectrum for FoMs:
            xP = pow(np.real(np.fft.fft(thisX)), 2)
            thisFftPowerXDotTrain[:, i] = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])

        # Change the data type away from complex if needed:
        if fftRegressionTarget != 'complex':
            thisFftXDotTrainTarget = thisFftXDotTrainTarget.astype(float)
        # Append:
        fftXDotTrainTargetAll.append(thisFftXDotTrainTarget)
        fftPowerXDotTrainAll.append(thisFftPowerXDotTrain)

        # FFT vectors for library functionals:
        thisFftLibFns = np.zeros((numFftPoints, thisL.shape[1]), dtype=complex)  # To regress against
                                                                            # fftXDotTrainTarget.
        for i in range(thisL.shape[1]):
            thisX = thisL[fomTimepointInds, i].copy()
            if functionList[i] == '1':
                fftTarget = np.zeros(numFftPoints)
            else:
                fftTarget = calculateFftForRegression_fn(thisX, numFftPoints, fftRegressionTarget)
            thisFftLibFns[:, i] = fftTarget

        # Change the data type away from complex if needed:
        if fftRegressionTarget != 'complex':
            thisFftLibFns = thisFftLibFns.astype(float)
        # Append:
        fftLibFnsAll.append(thisFftLibFns)

        # Generate histograms:
        thisXDotTrainHistogram = np.zeros((numHistogramBins, numVars))
        thisXDotTrainHistogramBins = np.zeros((numHistogramBins, numVars))
        for i in range(numVars):
            temp = np.histogram(thisXDotTrain[fomTimepointInds, i].flatten(), bins=numHistogramBins)
            thisXDotTrainHistogram[:, i] = temp[0]
            thisXDotTrainHistogramBins[:, i] = temp[1][0:-1]
        xDotTrainHistogramAll.append(thisXDotTrainHistogram)
        xDotTrainHistogramBinsAll.append(thisXDotTrainHistogramBins)

    #%% Ditto for xTest:
    # NOTE: These variables are currently not referred to again, since FoMs are not currently
    # done on xTest, and no regression is done on xTest.
    # So maybe this section can be refactored out.

    fftXDotTestTargetAll = []
    fftPowerXDotTestAll= []
    fftLibFnsTestAll= []
    xDotTestHistogramAll = []
    xDotTestHistogramBinsAll = []

    # for j in range(numTrajTest):
    #     thisXDotTest = xDotTestTrueAll[j]
    #     thisFftXDotTestTarget = np.zeros((numFftPoints, numVars), dtype=complex)  # truncated FFT,
    #     # used for regression
    #     thisFftPowerXDotTest = np.zeros((numFftPoints, numVars))  # FFT power spectrum, used for FoMs
    #     for i in range(numVars):
    #         # 1. Some form of FFT for regressions:
    #         thisX = thisXDotTest[:, i].copy()
    #         fftTarget = calculateFftForRegression_fn(thisX, numFftPoints, fftRegressionTarget)
    #         thisFftXDotTestTarget[:, i] = fftTarget
    #         # 2. Power spectrum for FoMs:
    #         xP = pow(np.real(np.fft.fft(thisX)), 2)
    #         thisFftPowerXDotTest[:, i] = xP[0:numFftPoints] / np.sum(xP[0:numFftPoints])
    #     if fftRegressionTarget != 'complex':
    #         thisFftXDotTestTarget = thisFftXDotTestTarget.astype(float)
    #     fftXDotTestTargetAll.append(thisFftXDotTestTarget)
    #     fftPowerXDotTestAll.append(thisFftPowerXDotTest)

    #     thisLTest = LTestAll[j]
    #     thisFftLibFnsTest = np.zeros((numFftPoints, thisLTest.shape[1]), dtype=complex)  # To regress
    #     # against.
    #     # fftXDotTrainTarget
    #     for i in range(thisLTest.shape[1]):
    #         thisX = thisLTest[:, i].copy()
    #         fftTarget = calculateFftForRegression_fn(thisX, numFftPoints, fftRegressionTarget)
    #         thisFftLibFnsTest[:, i] = fftTarget
    #     if fftRegressionTarget != 'complex':
    #         thisFftLibFnsTest = thisFftLibFns.astype(float)
    #     fftLibFnsTestAll.append(thisFftLibFnsTest)

    #     # Generate xDot histograms for FoMs:
    #     thisXDotTestHistogram = np.zeros((numHistogramBins, numVars))
    #     thisXDotTestHistogramBins = np.zeros((numHistogramBins, numVars))
    #     thisXDotTest = xDotTestTrueAll[j]
    #     for i in range(numVars):
    #         temp = np.histogram(thisXDotTest[:, i].flatten(), bins=numHistogramBins)  # range is
    #             # automatically set to (min, max)
    #         thisXDotTestHistogram[:, i] = temp[0]
    #         thisXDotTestHistogramBins[:, i] = temp[1][0:-1]
    #     xDotTestHistogramAll.append(thisXDotTestHistogram)
    #     xDotTestHistogramBinsAll.append(thisXDotTestHistogramBins)

    # The bulk of preparation work is now done.
    # Next, start the culling iterations.

    #%% Initializations for the fit-and-cull iterations:

    # Run algorithm on each training trajectory separately, using the other training trajectories for
    # "validation set" FoMs. Ideally this can be parallelized efficiently.
    # (During all iterations):
    # 1. Update coeffs with linear regression on multiple subsets of timepoints.
    # 2a. Cull variables and function types using reliability of coeff estimates.
    # 2b. Cull variables and function types using size of coeff estimates (weighted for fn value size).
    # Repeat 1-2 as necessary.

    # To store results on 'home' training trajectory:
    xTrainEvolvedAll = []
    xDotTrainPredictedAll = []  # results of a model on its training trajectory
    xDotTrainFirstSmoothedAll = []
    xDotTrainSmoothedAll = []

    # To store results on validation trajectories:
    indsValAll = []
    xValEvolvedAll = []
    xDotValComputedAll = []
    xDotValPredictedAll = []
    xDotValSmoothedAll = []

    # FoM storage. We make a list of lists, where each inner list corresponds to the results for one
    # training trajectory. For Val, the contents of each inner list is another list, of the FoMs for
    # each val trajectory.
    # Build the outer list here. Then the inner lists are populated for each training trajectory.
    historyCoeffArrayAll = [[] for i in range(numTrajTrain)]
    boundednessActionHistoryAll = [[] for i in range(numTrajTrain)]  # (whichIter, callSite, action)
    # triples recording what the boundedness constraint did. Empty if enforceBoundednessFlag.
    numBoundednessEigCallsAll = [0 for i in range(numTrajTrain)]
    historyProtectedFnsArrayAll = [[] for i in range(numTrajTrain)]
    historyWhichIterAll = [[] for i in range(numTrajTrain)]
    historyWhichCullAll = [[] for i in range(numTrajTrain)]
    historyInBoundsFoMAll = [[] for i in range(numTrajTrain)]
    historyInEnvelopeFoMAll = [[] for i in range(numTrajTrain)]
    historyStdDevFoMAll = [[] for i in range(numTrajTrain)]
    historyMeanFoMAll = [[] for i in range(numTrajTrain)]
    historyMedianFoMAll = [[] for i in range(numTrajTrain)]
    historyFftCorrelationFoMAll = [[] for i in range(numTrajTrain)]
    historyFftPowerAll = [[] for i in range(numTrajTrain)]
    historyHistogramCorrelationFoMAll = [[] for i in range(numTrajTrain)]
    historyHistogramsAll = [[] for i in range(numTrajTrain)]
    historyHistogramBinsAll = [[] for i in range(numTrajTrain)]
    historyMinHistCorrelationForEvolutionsAll = [[] for i in range(numTrajTrain)]
    historyXDotDiffEnvelopeAll = [[] for i in range(numTrajTrain)]
    historyXDotInBoundsFoMAll = [[] for i in range(numTrajTrain)]
    historyXDotHistogramCorrelationFoMAll = [[] for i in range(numTrajTrain)]
    historyXDotHistogramsAll = [[] for i in range(numTrajTrain)]
    historyXDotHistogramBinsAll = [[] for i in range(numTrajTrain)]

    # for xVal:
    historyCoeffArrayValAll = [[] for i in range(numTrajTrain)]  # to save coeff matrices of each step.
    historyProtectedFnsArrayValAll = [[] for i in range(numTrajTrain)]
    historyWhichIterValAll = [[] for i in range(numTrajTrain)]
    historyWhichCullValAll = [[] for i in range(numTrajTrain)]
    historyInBoundsFoMValAll = [[] for i in range(numTrajTrain)]
    historyInEnvelopeFoMValAll = [[] for i in range(numTrajTrain)]
    historyStdDevFoMValAll = [[] for i in range(numTrajTrain)]
    historyMeanFoMValAll = [[] for i in range(numTrajTrain)]
    historyMedianFoMValAll = [[] for i in range(numTrajTrain)]
    historyFftCorrelationFoMValAll = [[] for i in range(numTrajTrain)]
    historyFftPowerValAll = [[] for i in range(numTrajTrain)]
    historyHistogramCorrelationFoMValAll = [[] for i in range(numTrajTrain)]
    historyHistogramsValAll = [[] for i in range(numTrajTrain)]
    historyHistogramBinsValAll = [[] for i in range(numTrajTrain)]
    historyMinHistCorrelationForEvolutionsValAll = [[] for i in range(numTrajTrain)]
    historyXDotDiffEnvelopeValAll = [[] for i in range(numTrajTrain)]
    historyXDotInBoundsFoMValAll = [[] for i in range(numTrajTrain)]
    historyXDotHistogramCorrelationFoMValAll = [[] for i in range(numTrajTrain)]
    historyXDotHistogramsValAll = [[] for i in range(numTrajTrain)]
    historyXDotHistogramBinsValAll = [[] for i in range(numTrajTrain)]

    # Sentinels for the FoM dicts. They are only populated when a model actually gets evolved, and
    # they deliberately persist (stale) across iterations and trajectories once populated. Tracking
    # that with an explicit None is equivalent to, and much cheaper than, probing locals().
    fomDict = None
    fomDictVal = None

    #%% Loop through training trajectories, applying the training procedure to each trajectory
    # in turn. This could be parallelized, with care taken as to what order results get saved in to
    # the '*All' lists.
    for traj in range(numTrajTrain):

        print('\n' + '--------------------------------------------- \n' + \
            'Starting iterations on training trajectory ' + str(traj) + ': \n')

        #%% Initialize a bunch of variables to save results for this trajectory:
        whichIter = 0
        whichCull = 0 # to track iterations of cull-and-rerun.
        trajStartTime = time.time()  # for 'maxRunTimeSecs', reset per trajectory so every
        # trajectory gets the same wall-clock budget.
        iterateAgainFlag = True    # to start the while loop
        functionsToUseArray = functionsToUseArrayAll[traj]
        maxNumCulls = np.sum(functionsToUseArray.flatten())
        if restoreBasedOnFoMsFlag:
            maxNumCulls = int(1.3 * maxNumCulls)  # Allow a few extra.

        xTrain = xTrainAll[traj]
        pointWeights = pointWeightsAll[traj]
        L = LAll[traj]
        imputedSizeOfFunctionals = imputedSizeOfFunctionalsAll[traj]
        fftXDotTrainTarget = fftXDotTrainTargetAll[traj]
        fftLibFns = fftLibFnsAll[traj]
        simInitConds = simInitCondsAll[traj]
        xTrainFftPower = xTrainFftPowerAll[traj]
        varLocalMin = varLocalMinTrainAll[traj]
        varLocalMax = varLocalMaxTrainAll[traj]
        pointWeightsForFom = pointWeightsForFomAll[traj]
        xDotTrain = xDotTrainAll[traj]
        fftPowerXDotTrain = fftPowerXDotTrainAll[traj]
        xDotLocalMin = xDotLocalMinAll[traj]
        xDotLocalMax = xDotLocalMaxAll[traj]

        indsVal = np.array(range(numTrajTrain))
        indsVal = indsVal[indsVal != traj]
        xVal = [xTrainAll[i] for i in indsVal]
        simInitCondsVal = [simInitCondsAll[i] for i in indsVal]
        xValFftPower = [xTrainFftPowerAll[i] for i in indsVal]
        varLocalMinVal = [varLocalMinTrainAll[i] for i in indsVal]
        varLocalMaxVal = [varLocalMaxTrainAll[i] for i in indsVal]
        pointWeightsForFomVal = [pointWeightsForFomAll[i] for i in indsVal]
        xDotVal = [xDotTrainAll[i] for i in indsVal]
        fftPowerXDotVal = [fftPowerXDotTrainAll[i] for i in indsVal]
        xDotLocalMinVal = [xDotLocalMinAll[i] for i in indsVal]
        xDotLocalMaxVal = [xDotLocalMaxAll[i] for i in indsVal]

        # Initialize history arrays for this training trajectory:
        historyCoeffArray = []  # to save coeff matrices from each step.
        boundednessActionHistory = []  # what the boundedness constraint did, per call
        numBoundednessEigCalls = 0  # cost of the constraint, in eigendecompositions
        historyProtectedFnsArray = []
        historyWhichIter = []
        historyWhichCull = []
        historyInBoundsFoM = []
        historyInEnvelopeFoM = []
        historyStdDevFoM = []
        historyMeanFoM = []
        historyMedianFoM = []
        historyFftCorrelationFoM = []
        historyFftPower = []
        historyHistogramCorrelationFoM = []
        historyHistograms = []
        historyHistogramBins = []
        historyMinHistCorrelationForEvolutions = []
        historyXDotDiffEnvelope = []
        historyXDotInBoundsFoM = []
        historyXDotHistogramCorrelationFoM = []
        historyXDotHistograms = []
        historyXDotHistogramBins = []

        # for xVal:
        historyCoeffArrayVal= []  # to save coeff matrices from each step.
        historyProtectedFnsArrayVal = []
        historyWhichIterVal = []
        historyWhichCullVal = []
        historyInBoundsFoMVal = []
        historyInEnvelopeFoMVal= []
        historyStdDevFoMVal = []
        historyMeanFoMVal = []
        historyMedianFoMVal = []
        historyFftCorrelationFoMVal = []
        historyFftPowerVal = []
        historyHistogramCorrelationFoMVal = []
        historyHistogramsVal = []
        historyHistogramBinsVal = []
        historyMinHistCorrelationForEvolutionsVal = []
        historyXDotDiffEnvelopeVal = []
        historyXDotInBoundsFoMVal = []
        historyXDotHistogramCorrelationFoMVal = []
        historyXDotHistogramsVal = []
        historyXDotHistogramBinsVal = []

        indsValAll.append(indsVal)  # for easy recall

        # Initialize 'results' for this trajectory.
        # The keys must match the keys in 'fomChangesDict'.
        results = {'inBounds':historyInBoundsFoM,
                'evolutionsCorrelation':historyMinHistCorrelationForEvolutions,
                'histogramCorrelation':historyHistogramCorrelationFoM,
                'inEnvelopeFoM':historyInEnvelopeFoM,
                'fftPower':historyFftPower,
                'stdDevFoM':historyStdDevFoM}

        # Check that the dict keys match those specified in User Entries, if we are restoring 
        # functionals:    
        if results.keys() != fomChangesDict.keys() and restoreBasedOnFoMsFlag:
            print('Caution: results keys and fomChangesDict keys must match. Fix and restart.')
            
        # Make a counter. If a variable is culled, we want to start the process over with a restored
        # 'functionsToUseArray' (except for functional that use the culled variable):
        numRemovedVariables = 0

        preCullFunctionsToUseArray = functionsToUseArray  # Initialize
        protectedFunctionsArray = np.zeros(functionsToUseArray.shape, dtype=int)  # initialize
        # Permantently protect any indicated functionals:
        if permanentlyProtectedFunctions.shape[0] > 0:
            protectedFunctionsArray[permanentlyProtectedFunctions] = 1000
        restorableFnsArray = np.zeros(functionsToUseArray.shape, dtype=bool)
        # Functionals that the boundedness constraint had to put back into the library. They are
        # exempted from both the in-span cull and the coefficient-size cull below, either of which
        # would otherwise remove them again on the next iteration and livelock. Note that
        # 'protectedFunctionsArray' cannot serve here: the in-span cull does not consult it at all,
        # and the coefficient-size cull consults it only in its ratcheting-threshold step, not in
        # the too-small step that does the damage.
        boundednessRequiredFnsArray = np.zeros(functionsToUseArray.shape, dtype=bool)

        # Initialize the threshold to increment:
        cullableFunctionFlag = False
        restoreFnFlag = False
        skipCullByCoeffSizeDueToInSpanCullFlag = False

        modelActiveLib = functionsToUseArray
        modelCoefs = functionsToUseArray.astype(float)

        # Initializations are done.

        #%% Now start iterations for this trajectory:
        while iterateAgainFlag == True:

            # decrement the counts of protected iterations per {var, fn}:
            protectedFunctionsArray -= 1
            protectedFunctionsArray[protectedFunctionsArray < 0] = 0

            console = sys.stdout
            with open(outputFilename, 'a') as file:
                print('--------------------------------------------', file=file)
                print('Iteration ' + str(whichIter) + ', trajectory ' + str(traj) + ':', file=file)
                sys.stdout = console
                file.close()

            #######################################################

            #%% 1. Regress (each iteration).
            # If 'basic' lin reg, this will determine only the coeffs that go into
            # assessWhetherToCull...(), not the eligible-for-cull fns.
            # If 'ridge' regr, eligible-for-cull functions may change as well. Note that ridge
            # regression does not work as well, so it can perhaps be refactored out.
            # Coefficient estimates are typically much better after regression. These coefficients
            # then get examined by the culling function.
            # To regress, we also need to have culled some new function (2nd condition in 'if' below).

            postRegressionCoeffs = modelCoefs * modelActiveLib
            # These coeffs are from the previous regression (from the last cullAndRerun iteration),
            # with coeffs for any newly culled functions zeroed out. We'll generate new coefficients
            # in this iteration.

            # Alias xTrain, for ease. Read-only below (shape queries and column reads passed to
            # estimateDerivatives_fn), so no copy is needed:
            xT = xTrain

            #%% 1a. Define some timepoints to regress on, 'startIndsToUse', so-called because they are
            # the first timepoint to use in eg euler or 4th order approximation of the derivative.
            # Perhaps remove any timepoints where the active functionals have wildly big ratios.

            if parseTimepointsByVarValuesFlag:  # Case (but True works best): Ignore timepoints where
            # the difference between library functions is too great ('maxFnValRatio') for too many
            # library functions. Done separately for each variable, since they have different active
            # libraries.
                startIndsToUse = \
                    parseTimepointsByMagnitudesOfVariables_fn(L, functionsToUseArray, 
                                                            nonConstantIndices, margin, 
                                                            minNumStartIndsToUse, maxFnValRatio)
                if printDiagnosticsForParsingFlag:
                    console = sys.stdout
                    with open(outputFilename, 'a') as file:
                        print('\n' + 'numTimepoints for each var = ' + \
                        str(np.sum(startIndsToUse,axis=1).flatten()), file=file)
                        sys.stdout = console
                        file.close()

            else:  # Case: random selection of points, without constraint, to regress at (de facto
            # may be almost all points). Note: This case is not currently used.
                numSamplesToDraw = min(len(fomTimepointInds), len(tTrain) - 2*margin)
                startInds = np.sort(np.random.choice(range(margin, len(tTrain) - margin), 
                                                    numSamplesToDraw, replace = False))
                startIndsToUse = np.zeros((xT.shape[0] - 2*margin, numVars), dtype=bool)
                for v in range(startIndsToUse.shape[1]):
                    startIndsToUse[startInds, v] = True

            # 'startIndsToUse' is an array of booleans, showing timepoints to use, one column per
            # variable. The first entry is for timepoint 'margin', and the last is for 
            # {len(tTrain) - margin}.

            # Note that the constant functional is in our library. So we don't want the regression
            # to add an intercept term, hence fitInterceptFlag = False in the calls below.
            fitInterceptFlag = False

            #%% 1b. Regress on each variable using each segment in turn.
            # So do numVars * numRegressionSegments regressions. For each variable, each active
            # functional will end up with numRegressionSegments coefficient estimates.
            # Then compare and combine these sets of coefficients (roughly, take their median).

            # Initialize storage for new coefficients:
            postRegressionCoeffsBySeg = -100 * np.ones((len(variableNames), len(functionList),
                                                        numRegressionSegments))

            # Storage for the full (all-timepoints) design matrices, used only by the boundedness
            # constraint below, which needs to refit rows outside the segment structure:
            designMatrices = {}

            # 1b(i) Removed in abridged wrapper.

            # The outer loop is over variables, the inner loop is over segments.
            for v in range(len(variableNames)):

                pointWtsForThisVar = pointWeights[:, v]

                if np.sum(functionsToUseArray[v, :]) == 0:  # Case: this variable has no usable
                # functionals in its library.
                    pass  # postRegressionCoeffs[v, :] remains equal to 0

                else:  # Case: This variable has a non-zero library. Do regressions on each segment:
                    # Convert booleans to indices of timepoints:
                    startIndsVarI = np.where(startIndsToUse[v, ])[0] + margin

                    # These three are the same for every segment, so compute them once here rather
                    # than inside the segment loop:
                    # Clamp to the number of available timepoints, since we sample without
                    # replacement below. The clamp binds only if
                    # (1 + 2*overlapFraction) >= numRegressionSegments, which is flagged above:
                    numPointsPerSegment = \
                        min(len(startIndsVarI),
                            int(len(startIndsVarI) / numRegressionSegments * \
                                (1 + 2 * overlapFraction)))
                    functionInds = np.where(functionsToUseArray[v, :] == True)[0]
                    xTThisVar = xT[:, v]

                    # Cache this variable's design matrix over *all* its timepoints (the segment
                    # loop below uses random subsets). 'stabilizeLinearModel_fn' refits on it when
                    # the model violates the boundedness constraint. One extra derivative estimate
                    # per variable per iteration, which is negligible beside the solve_ivp calls
                    # that dominate the profile. NOTE: this consumes no random numbers, so it does
                    # not perturb the segment sampling below:
                    if enforceBoundednessFlag and len(startIndsVarI) > 0 and len(functionInds) > 0:
                        yFull, weightsFull = \
                            estimateDerivatives_fn(xTThisVar, startIndsVarI, numDtStepsForDeriv,
                                                pointWtsForThisVar, dt)
                        if not weightTimepointsFlag:
                            weightsFull = np.ones(weightsFull.shape) / len(weightsFull)
                        designMatrices[v] = {'X': L[np.ix_(startIndsVarI, functionInds)],
                                            'y': yFull,
                                            'sampleWeights': weightsFull,
                                            'functionInds': functionInds}

                    for seg in range(numRegressionSegments):
                        # 1b(ii) Pick a random subset of startIndsVarI for this segment:
                        startIndIndices = np.sort(np.random.choice(range(len(startIndsVarI)),
                                                numPointsPerSegment, replace=False))
                        startInds = startIndsVarI[startIndIndices]


                        # Check that we have both valid startInds and active functionals:
                        if len(startInds) > 0 and len(functionInds) > 0:  # Case: there are some
                        # startInds in this segment and some functionals left for this variable.

                            # 1b(iii) Define the regression target and timepoint weights:
                            y, weights = \
                                estimateDerivatives_fn(xTThisVar, startInds, numDtStepsForDeriv,
                                                    pointWtsForThisVar, dt)
                            # Extract the relevant functions (rows and columns in one gather, to
                            # avoid materializing all library columns at the chosen timepoints):
                            X = L[np.ix_(startInds, functionInds)]

                            if weightTimepointsFlag:
                                sampleWeights = weights
                            else:
                                sampleWeights = np.ones(weights.shape) / len(weights)  # uniform, sum
                                # to 1

                            # Also perhaps include fft(xDotTrain) as targets. There are many
                            # fewer regression points in the fft target: 50 vs. up to 5000 for the
                            # usual raw derivs, so use the sample weights to balance this out.
                            if regressOnFftAlpha > 0:
                                rawYAlpha = 1 - regressOnFftAlpha
                                yFftOfXDot = fftXDotTrainTarget[:, v]
                                y = np.hstack((y, yFftOfXDot ))  # Stack the targets
                                XForFft = fftLibFns[:, functionInds].copy()
                                XForFft[np.where(np.isnan(XForFft))] = 0  # since fft of 1 == nan
                                X = np.vstack((X, XForFft))

                                sampleWeights = \
                                    np.hstack((rawYAlpha * sampleWeights, regressOnFftAlpha * \
                                            np.ones(numFftPoints) / numFftPoints))

                            # 1b(iv) Finally ready to do the regression:
                            # Special case: If we are regressing on complex FFT, we need to use lstsq:
                            if fftRegressionTarget == 'complex' and regressOnFftAlpha > 0:
                                w = sampleWeights.reshape(-1, 1)
                                betas = lstsq(np.tile(np.sqrt(w), (1, X.shape[1])) * X,
                                            np.sqrt(w) * y.reshape(-1, 1), rcond=-1)[0]
                                if whichCull > 20 and seg == 0:
                                    print('whichIter = ' + str(whichIter) + ' ' + variableNames[v] + \
                                        ', betas = ' + str(np.round(betas, 2).transpose()))
                                betas = np.real(betas)  # NOTE: This may be important and harmful.
                                postRegressionCoeffsBySeg[v, functionInds, seg] = betas.flatten()
                            else: # case: all real values in X and y
                                betas = weightedLeastSquares_fn(X, y, sampleWeights,
                                                                fitInterceptFlag)[0]
                                postRegressionCoeffsBySeg[v, functionInds, seg] = betas

            # The 3-D postRegressionCoeffsBySeg array is now populated, with meaningful values and/or
            # -100s. Regressions are done for all variables and segments.

            #%% 2. Process the collections of coeffs from the different segments to get a single set of
            # coeffs, as follows: For each {variable, library function} pair, if
            # stdDev(coeffs) / median(coeffs) is low, set coeff = median. Else remove the biggest
            # outlier and repeat. This is a fancier version of taking the median.
            # The main work is done in 'combineSegmentCoeff_fn'

            postRegressionStdOverMedian = np.zeros(postRegressionCoeffs.shape)  # Initialize an array
            # to contain noisiness values for each {variable, function} pair. Irrelevant if we do not
            # cull based on noisiness of coefficient estimates.

            prs = postRegressionCoeffsBySeg
            for v in range(len(variableNames)):
                # Depends only on v, so evaluate it once per variable rather than per functional:
                printDiagnosticsFlag = \
                    np.sum(prs[v, :, 0] != -100) <= printCoeffsForEachSegmentThresh and \
                        v < 4 # ie few enough functionals in this variable, and few enough
                        # variables overall, that we can print the outcome.
                for j in range(len(functionList)):
                    if functionsToUseArray[v, j]:  # Case: this functional is in use for this var (it
                    # has not been culled), so we need to calculate a single coeff.

                        # Decide what coeff value to carry forward:
                        coeff, stdOverMedian = \
                            combineSegmentCoeffs_fn(prs[v, j, :], printDiagnosticsFlag,
                                                    variableNames[v], functionList[j], snrThreshold, 
                                                    minNumSegmentResultsToUse, outputFilename)

                        # Assign the final coeffs for this {variable, function} pair:
                        postRegressionCoeffs[v, j] = coeff
                        postRegressionStdOverMedian[v, j] = stdOverMedian # In case we cull functions
                        # based on high variability.

            # postRegressionCoeffs and postRegressionStdOverMedian are now both complete.

            #%% 2b. Constrained optimization (trapping SINDy, linear case): force the model to
            # produce bounded trajectories. The regressions above are done one variable at a time,
            # but the constraint sym(A) <= gamma*I couples all the rows, so it can only be applied
            # here, to the assembled coefficient matrix. See 'stabilizeLinearModel_fn'.
            if enforceBoundednessFlag:
                preStabilizationLib = functionsToUseArray.copy()
                postRegressionCoeffs, functionsToUseArray, boundednessAction, numEigCalls = \
                    enforceBoundedness_fn(postRegressionCoeffs, functionsToUseArray,
                                          designMatrices)
                boundednessActionHistory.append((whichIter, 'postRegression', boundednessAction))
                numBoundednessEigCalls += numEigCalls
                # Remember anything the constraint put back, so the culls below leave it alone:
                boundednessRequiredFnsArray = np.logical_or(
                    boundednessRequiredFnsArray,
                    np.logical_and(functionsToUseArray, np.logical_not(preStabilizationLib)))

            #%% Book-keeping section:
            # Now all variables have been fitted (maybe twice if we culled based on std/mean) and
            # have new coeffs assigned. Replace the coeffs of the current model:
            preStabilizationActiveLib = functionsToUseArray.copy()  # to detect functionals lost to
            # the shrinkage of the ridge step, just below
            functionsToUseArray = np.abs(postRegressionCoeffs) > almostZeroRoundDownThreshold
            modelActiveLib = functionsToUseArray
            modelCoefs = postRegressionCoeffs * functionsToUseArray  # To avoid weird
            # "almost zero" coefs

            # The ridge step of the boundedness constraint shrinks coefficients toward the
            # projected target, so it can push an active coeff below the threshold just applied,
            # culling it as a side effect. Log that rather than let it show up as a mystery in the
            # coefficient history:
            if enforceBoundednessFlag:
                numShrunkOut = int(np.sum(np.logical_and(preStabilizationActiveLib,
                                                        np.logical_not(functionsToUseArray))))
                if numShrunkOut > 0:
                    with open(outputFilename, 'a') as file:
                        print('Boundedness constraint (' + boundednessAction + ') shrank ' + \
                            str(numShrunkOut) + ' coeff(s) below almostZeroRoundDownThreshold, ' + \
                            'so they were culled.', file=file)

            # Print current model:
            this = printModel_fn(modelCoefs, variableNames, functionList)
            console = sys.stdout
            with open(outputFilename, 'a') as file:
                print('\n' + ' Linear Regression (' + \
                    str(numDtStepsForDeriv) + '*dt), target = estimated derivatives ' + \
                        ', on results of whichIter = ' + str(whichIter) + '\n' + this + '\n',
                        file=file)
                sys.stdout = console
                file.close()

            # Diagnostic: If wished, print a version with weighted coeffs:
            if printWeightedCoefficientModelFlag:
                if coeffWeightsCutoffFactor > 1:  # Recall coeffWeightsCutoffFactor = 1 -> no weights.
                    weightArray = \
                        calcWeightArray_fn(modelCoefs, modelActiveLib,
                                        imputedSizeOfFunctionals, coeffWeightsCutoffFactor, 
                                        percentileOfImputedValuesForWeights)
                else:  # Case: we're not weighting the coeffs.
                    weightArray = modelActiveLib.astype(int)
                printWeightedCoeffModel_fn(modelCoefs, functionsToUseArray, weightArray, 
                                        variableNames, functionList, outputFilename)

            # To catch nans in coeffs:
            modelCoefs[np.where(np.isnan(modelCoefs))] = 0

            #%% 3. Calculate Figures of Merit for this iteration, post-regression (pre-cull):
            # First evolve the data from an estimated initial condition. Then calculate:
            # 1. percentage of time each evolved variable stays within its noise envelope;
            # 2. percentage of time each evolved variable stays within its max-min bounds;
            # 3, 4, 5. std dev, mean, median of each evolved time-series;
            # 6. fft power spectrum correlation; 7. histogram correlation;
            # 8. stability of evolutions, by doing multiple evolutions and comparing histograms);
            # 9. comparing xDot evolutions. We want these generally close but not exact (since the LP
            #    hamming filter gives incorrect derivatives).

            # print('whichIter = ' + str(whichIter) + ', simInitConds = ' + \
            #       str(np.round(simInitConds, 2)))

            # Define the time window we wish to test for matching:
            fomTimepoints = tTrain[fomTimepointInds]

            # 3a. Calculate FoMs for xTrain, UNLESS we culled based on in-span (ie culled using 
            # linear dependence of active functionals), in which case we skip the FoMs (to avoid the
            # cost of time-series evolutions):
            if (skipCullByCoeffSizeDueToInSpanCullFlag and doNotEvolveAfterInSpanCullFlag) or \
                max(np.sum(functionsToUseArray, axis=1)) > maxNumFunctionalsToEvolve:
                print('iter ' + str(whichIter) + \
                    '. Skip FoM evolutions because we culled an in-span functional. \n')
            else:
                print('iter ' + str(whichIter) + '. Evolving models for FoMs...\n')
                # If doing multiple evolutions (to check model stability) there's some extra work:
                if numEvolutionsForTrainFom > 1: # Case: we're doing multiple evolutions of the model
                # to test for model stability. Maybe use parallel processing.
                    localHistograms = []
                    numEvolutionsDone = 0
                    preTime = time.time()
                    while numEvolutionsDone < numEvolutionsForTrainFom:
                        # Evolve this model over just the FoM timepoints:
                        xTrainEvolved = evolveModel_fn(modelCoefs, recipes,
                                                    simInitConds.copy(), fomTimepoints)
                        # Calculate the FoMs to get the histogram for this evolution:
                        fomDict = \
                            calculateFiguresOfMerit_fn(xTrainEvolved, xTrain[fomTimepointInds, :],
                                                    xTrainFftPower, varLocalMin, varLocalMax,
                                                    stdDevTrainAll[traj], meanTrainAll[traj],
                                                    medianTrainAll[traj], fomTimepoints,
                                                    maxPhaseShift)
                        localHistograms.append(fomDict['histograms'])
                        numEvolutionsDone += 1

                    postTime = time.time()


                    # Note: The evolution that will be used for most of the FoMs is the last one
                    # that occurred (the final version of 'fomDict').
                    # If we did multiple evolutions, compare their histograms. Our goal is to see if
                    # the evolutions diverged from each other (stability). All the bins (for each 
                    # variable) are the same due to 'calculateFiguresOfMerit_fn'.
                    minHistCorr = 10*np.ones(numVars)
                    this = localHistograms[0]    # numBins x numVars
                    for i in range(1, numEvolutionsForTrainFom):
                        that = localHistograms[i]
                        for j in range(numVars):
                            histCorr = np.dot(this[:, j], that[:, j]) / np.dot(this[:, j], this[:, j])
                            minHistCorr[j] = min(minHistCorr[j], histCorr)
                else:  # case: We do just one evolution (so we're not testing for model stability)
                    
                    minHistCorr = np.ones(numVars)
                    xTrainEvolved = evolveModel_fn(modelCoefs, recipes, simInitConds.copy(), 
                                                fomTimepoints)
                    fomDict = calculateFiguresOfMerit_fn(xTrainEvolved, xTrain[fomTimepointInds, :],
                                                        xTrainFftPower, varLocalMin, varLocalMax,
                                                        stdDevTrainAll[traj], meanTrainAll[traj],
                                                        medianTrainAll[traj], fomTimepoints, 
                                                        maxPhaseShift)

                # 3b. xDot FoMs: These do not depend on a particular evolution, just on the
                # coefficients of the current model:
                xDotTrainPredicted = \
                    calculateDerivativesFromModel_fn(xTrain[fomTimepointInds, :], modelCoefs,
                                                    L[fomTimepointInds, :])
                # (i) Compare this iteration's xDot predictions to the "true" via the 80th %ile of
                # their difference, normalized:
                dummyForXDotFom = np.ones(meanTrainAll[traj].shape) # dummy argin for xDot calcFoM_fn
                maxXDotVals = np.max(np.abs(xDotTrain[fomTimepointInds, :]), axis=0)  # for normalizing
                xDotDiff = xDotTrain[fomTimepointInds, :] - xDotTrainPredicted
                xDotDiffEnvelope = np.zeros(numVars)
                for i in range(numVars):
                    xDotDiffEnvelope[i] = np.percentile(np.abs(xDotDiff[:, i]), 80) / maxXDotVals[i]

                # (ii) More xDot FoMs:
                xDotFomDict = \
                    calculateFiguresOfMerit_fn(xDotTrainPredicted, xDotTrain[fomTimepointInds, :],
                                            fftPowerXDotTrain, xDotLocalMin, xDotLocalMax,
                                            dummyForXDotFom, dummyForXDotFom, dummyForXDotFom, 
                                            fomTimepoints, 1)

            # 3c. Save the FoMs from this iteration (using FoMs of the last of the evolutions).
            # If we culled an in-span functional, all the FoMs except number of functionals will
            # repeat from the previous time.

            # Special case: If we have been skipping evolutions at high functional counts (to avoid
            # hanging), 'fomDict' may not yet be defined. In this case, define it here:
            if fomDict is None:
                dummyVal = -1 * np.ones((1, numVars))
                fomDict = dict()
                fomDict['inBoundsFoM'] = dummyVal
                fomDict['inEnvelopeFoM'] = dummyVal
                fomDict['stdDevFoM'] = dummyVal
                fomDict['meanFoM'] = dummyVal
                fomDict['medianFoM'] = dummyVal
                fomDict['fftCorrelationFoM'] = dummyVal
                fomDict['fftPower'] = dummyVal
                fomDict['histogramCorrelationFoM'] = dummyVal
                fomDict['histograms'] = dummyVal
                fomDict['histogramBins'] = dummyVal
                minHistCorr = dummyVal
                xDotFomDict = dict()
                xDotFomDict['inBoundsFoM'] = dummyVal
                xDotFomDict['histogramCorrelationFoM'] = dummyVal
                xDotFomDict['histograms'] = dummyVal
                xDotFomDict['histogramBins'] = dummyVal
                xDotDiffEnvelope = dummyVal

            # Now save this iter's foms:
            historyWhichIter.append(whichIter)
            historyWhichCull.append(whichCull)
            historyCoeffArray.append(modelCoefs.copy())  # also gives count of active fnals
            historyInBoundsFoM.append(fomDict['inBoundsFoM'])
            historyInEnvelopeFoM.append(fomDict['inEnvelopeFoM'])
            historyStdDevFoM.append(fomDict['stdDevFoM'])
            historyMeanFoM.append(fomDict['meanFoM'])
            historyMedianFoM.append(fomDict['medianFoM'])
            historyFftCorrelationFoM.append(fomDict['fftCorrelationFoM'])
            historyFftPower.append(fomDict['fftPower'])
            historyHistogramCorrelationFoM.append(fomDict['histogramCorrelationFoM'])
            historyHistograms.append(fomDict['histograms'])
            historyHistogramBins.append(fomDict['histogramBins'])
            historyMinHistCorrelationForEvolutions.append(minHistCorr)
            historyXDotDiffEnvelope.append(xDotDiffEnvelope)
            historyXDotInBoundsFoM.append(xDotFomDict['inBoundsFoM'])
            historyXDotHistogramCorrelationFoM.append(xDotFomDict['histogramCorrelationFoM'])
            historyXDotHistograms.append(xDotFomDict['histograms'])
            historyXDotHistogramBins.append(xDotFomDict['histogramBins'])

            #----------------------------------------------------------
            # 3d. Calculate FoMs for xVal, for each val trajectory in turn. An FoM from each val
            # trajectory get saved in a list, which is then appended to the relevant history List:
            # Note: This section largely repeats the section for FoMs on xTrain, but further
            # refactoring would be a nuisance.
            localInBoundsFoMVal = []  # 'local...' because it is for this iteration only
            localInEnvelopeFoMVal = []
            localStdDevFoMVal = []
            localMeanFoMVal = []
            localMedianFoMVal = []
            localFftCorrelationFoMVal = []
            localFftPowerVal = []
            localHistogramCorrelationFoMVal = []
            localHistogramsVal = []
            localHistogramBinsVal = []
            localMinHistCorrelationForEvolutionsVal = []
            localXDotDiffEnvelopeVal = []
            localXDotInBoundsFoMVal = []
            localXDotHistogramCorrelationFoMVal = []
            localXDotHistogramsVal = []
            localXDotHistogramBinsVal = []

            # Same for every val trajectory, so evaluate the skip test once:
            skipValEvolutionsFlag = \
                skipCullByCoeffSizeDueToInSpanCullFlag and doNotEvolveAfterInSpanCullFlag or \
                max(np.sum(functionsToUseArray, axis=1)) > maxNumFunctionalsToEvolve

            for k in range(numTrajTrain - 1):
                xVal = xTrainAll[indsVal[k]]  # we will evolve over timepointsForFoMs
                LVal = LAll[indsVal[k]]  # values of the functionals on this trajectory
                initConds = simInitCondsAll[indsVal[k]]
                varLocalMinVal = varLocalMinTrainAll[indsVal[k]]
                varLocalMaxVal = varLocalMaxTrainAll[indsVal[k]]
                xValFftPower = xTrainFftPowerAll[indsVal[k]]
                #xValInitCond = simInitCondsAll[indsVal[k]]  # this starts at 'margin'
                thisStdDev = stdDevTrainAll[indsVal[k]]
                thisMean = meanTrainAll[indsVal[k]]
                thisMedian = medianTrainAll[indsVal[k]]
                xDotVal = xDotTrainAll[indsVal[k]]
                xDotLocalMaxVal = xDotLocalMaxAll[indsVal[k]]
                xDotLocalMinVal = xDotLocalMinAll[indsVal[k]]
                fftPowerXDotVal = fftPowerXDotTrainAll[indsVal[k]]
                dummyForXDotFom = np.ones(thisMean.shape)  # dummy argin for xDot calcFoM_fn

                if skipValEvolutionsFlag:
                    pass
                else:
                    # 3d(i) Evolve val trajectories or trajectory, for FoMs:
                    if numEvolutionsForValFom > 1:
                        localHistograms = []
                        numEvolutionsDone = 0
                        while numEvolutionsDone < numEvolutionsForValFom:
                            # Evolve this model over just the FoM timepoints:
                            xValEvolved = evolveModel_fn(modelCoefs, recipes, initConds,
                                                        fomTimepoints)
                            xDotValPredicted = \
                                calculateDerivativesFromModel_fn(xVal[fomTimepointInds, :],
                                                                modelCoefs,
                                                                LVal[fomTimepointInds, :])

                            # Calculate and print the FoMs:
                            fomDictVal = \
                                calculateFiguresOfMerit_fn(xValEvolved, xVal[fomTimepointInds, :],
                                                        xValFftPower, varLocalMinVal,
                                                        varLocalMaxVal, thisStdDev, thisMean,
                                                        thisMedian, fomTimepoints, 
                                                        maxPhaseShift)
                            localHistograms.append(fomDict['histograms'])
                            numEvolutionsDone += 1

                        # Compare histograms of the different evolutions: All the bins (for each
                        # variable) are the same due to 'calculateFiguresOfMerit_fn'.
                        minHistCorrVal = 10*np.ones(numVars)
                        this = localHistograms[0]    # numBins x numVars
                        for i in range(1, numEvolutionsForValFom):
                            that = localHistograms[i]
                            for j in range(numVars):
                                histCorrVal = np.dot(this[:, j],
                                                    that[:, j]) / np.dot(this[:, j], this[:, j])
                                minHistCorrVal[j] = min(minHistCorrVal[j], histCorrVal)  # the FoM
                    else:  # case: We're doing just one evolution (not checking model stability)
                        minHistCorrVal = np.ones(numVars)  # set the stability FoM == 1
                        # Do one evolution of xVal to calculate FoMs:
                        xValEvolved = evolveModel_fn(modelCoefs, recipes, initConds, fomTimepoints)

                    # 3d(ii) Calculate and print the FoMs:
                    fomDictVal = \
                        calculateFiguresOfMerit_fn(xValEvolved, xVal[fomTimepointInds, :],
                                                xValFftPower, varLocalMinVal, varLocalMaxVal,
                                                thisStdDev, thisMean, thisMedian, fomTimepoints, 
                                                maxPhaseShift)
                    # 3d(iii). xDot FoMs on Validation:
                    xDotValPredicted = \
                        calculateDerivativesFromModel_fn(xVal[fomTimepointInds, :], modelCoefs,
                                                        LVal[fomTimepointInds, :])
                    maxXDotVals = np.max(np.abs(xDotVal), axis=0)  # for normalizing
                    xDotDiffVal = xDotVal[fomTimepointInds, :] - xDotValPredicted
                    xDotDiffEnvelopeVal = np.zeros(numVars)
                    for i in range(numVars):
                        xDotDiffEnvelopeVal[i] = \
                            np.percentile(np.abs(xDotDiffVal[:, i]), 80) / maxXDotVals[i]
                    xDotFomDictVal = \
                        calculateFiguresOfMerit_fn(xDotValPredicted, xDotVal[fomTimepointInds, :],
                                                fftPowerXDotVal, xDotLocalMinVal, xDotLocalMaxVal,
                                                dummyForXDotFom, dummyForXDotFom, dummyForXDotFom,
                                                fomTimepoints, 1)
                # 3d(iv). Save FoMs for this val trajectory:
                if fomDictVal is None:
                    dummyVal = -1 * np.ones((1, numVars))
                    fomDictVal = dict()
                    fomDictVal['inBoundsFoM'] = dummyVal
                    fomDictVal['inEnvelopeFoM'] = dummyVal
                    fomDictVal['stdDevFoM'] = dummyVal
                    fomDictVal['meanFoM'] = dummyVal
                    fomDictVal['medianFoM'] = dummyVal
                    fomDictVal['fftCorrelationFoM'] = dummyVal
                    fomDictVal['fftPower'] = dummyVal
                    fomDictVal['histogramCorrelationFoM'] = dummyVal
                    fomDictVal['histograms'] = dummyVal
                    fomDictVal['histogramBins'] = dummyVal
                    minHistCorrVal = dummyVal
                    xDotFomDictVal = dict()
                    xDotFomDictVal['inBoundsFoM'] = dummyVal
                    xDotFomDictVal['histogramCorrelationFoM'] = dummyVal
                    xDotFomDictVal['histograms'] = dummyVal
                    xDotFomDictVal['histogramBins'] = dummyVal
                    xDotDiffEnvelopeVal = dummyVal

                localInBoundsFoMVal.append(fomDictVal['inBoundsFoM'])
                localInEnvelopeFoMVal.append(fomDictVal['inEnvelopeFoM'])
                localStdDevFoMVal.append(fomDictVal['stdDevFoM'])
                localMeanFoMVal.append(fomDictVal['meanFoM'])
                localMedianFoMVal.append(fomDictVal['medianFoM'])
                localFftCorrelationFoMVal.append(fomDictVal['fftCorrelationFoM'])
                localFftPowerVal.append(fomDictVal['fftPower'])
                localHistogramCorrelationFoMVal.append(fomDictVal['histogramCorrelationFoM'])
                localHistogramsVal.append(fomDictVal['histograms'])
                localHistogramBinsVal.append(fomDictVal['histogramBins'])
                localMinHistCorrelationForEvolutionsVal.append(minHistCorrVal)
                localXDotDiffEnvelopeVal.append(xDotDiffEnvelopeVal)
                localXDotInBoundsFoMVal.append(xDotFomDictVal['inBoundsFoM'])
                localXDotHistogramCorrelationFoMVal.append(xDotFomDictVal['histogramCorrelationFoM'])
                localXDotHistogramsVal.append(xDotFomDictVal['histograms'])
                localXDotHistogramBinsVal.append(xDotFomDictVal['histogramBins'])

            # 3d(v). All Validation trajectories now have FoMs. Save Val info from this iteration:
            historyWhichIterVal.append(whichIter)
            historyWhichCullVal.append(whichCull)
            historyCoeffArrayVal.append(modelCoefs.copy())
            historyInBoundsFoMVal.append(localInBoundsFoMVal)
            historyInEnvelopeFoMVal.append(localInEnvelopeFoMVal)
            historyStdDevFoMVal.append(localStdDevFoMVal)
            historyMeanFoMVal.append(localMeanFoMVal)
            historyMedianFoMVal.append(localMedianFoMVal)
            historyFftCorrelationFoMVal.append(localFftCorrelationFoMVal)
            historyFftPowerVal.append(localFftPowerVal)
            historyHistogramCorrelationFoMVal.append(localHistogramCorrelationFoMVal)
            historyHistogramsVal.append(localHistogramsVal)
            historyHistogramBinsVal.append(localHistogramBinsVal)
            historyMinHistCorrelationForEvolutionsVal.append(localMinHistCorrelationForEvolutionsVal)
            historyXDotDiffEnvelopeVal.append(localXDotDiffEnvelopeVal)
            historyXDotInBoundsFoMVal.append(localXDotInBoundsFoMVal)
            historyXDotHistogramCorrelationFoMVal.append(localXDotHistogramCorrelationFoMVal)
            historyXDotHistogramsVal.append(localXDotHistogramsVal)
            historyXDotHistogramBinsVal.append(localXDotHistogramBinsVal)

            # Note: we are now done using 'modelForFom' (until next iteration).

            #%% 4. Restore culled functionals:
            # Compare current and past values of selected FoMs. If there is a triggering drop from a
            # sufficiently strong value, and the evolutions all agreed, restore the culled
            # functional and protect it for some future iterations.
            # current dict = {'inBounds':0.8, 'evolutionsCorrelation':0.8, 'histogramCorrelation':0.8}
            restoreFnFlag = False  # initialize
            evolThreshold = fomChangesDict['evolutionsCorrelation'][1]
            if len(historyWhichCull) > 1 and restoreBasedOnFoMsFlag and \
                np.sum(restorableFnsArray.flatten()) > 0 and whichIter > startAfterWhichIter and \
                    np.min(historyMinHistCorrelationForEvolutions[-2]) > evolThreshold:
                fomsToMonitor = fomChangesDict.keys()
                # 4a. For each monitored FoM, check whether there was a triggering drop:
                for key in fomsToMonitor:
                    this = results[key]
                    fomChangeThresh = fomChangesDict[key][0]
                    fomMinPreviousVal = fomChangesDict[key][1]
                    if key != 'stdDevFoM': # ie most foms
                        restoreFnFlag = (np.min(this[-2]) > fomMinPreviousVal and \
                                        np.min(this[-1] / this[-2]) < fomChangeThresh) or \
                                        restoreFnFlag
                    if key == 'stdDevFoM': # use 1 - clipped(abs(std dev value)):
                        temp = np.abs(this[-2])
                        for i in range(len(temp)):
                            temp[i] = min(1, temp[i])
                        adjPreviousVal = 1 - temp
                        temp = np.abs(this[-1])
                        for i in range(len(temp)):
                            temp[i] = min(1, temp[i])
                        adjCurrentVal = 1 - temp
                        restoreFnFlag = \
                            (np.min(adjPreviousVal) > fomMinPreviousVal) and \
                            (np.min(adjPreviousVal - adjCurrentVal) > fomChangeThresh) or \
                            restoreFnFlag  # ie both of first two conditions OR 'restoredFnFlag'
                        # use np.min() because 'this' is a vector, with length = numVars

                #  4b. Restore and protect functionals if indicated:
                if restoreFnFlag:
                    restoredFnStr = ''
                    for i in np.where(np.sum(restorableFnsArray, axis=1) > 0)[0]:  # ie rows/variables
                    # with restored functionals.
                        restoredFnStr = restoredFnStr + variableNames[i] + ': ' + \
                            str(np.array(functionList)[restorableFnsArray[i, :]]) + ' '
                    if len(restoredFnStr) > 0:
                        print('iter ' + str(whichIter) + '. Restored functionals: ' + restoredFnStr)
                    console = sys.stdout
                    with open(outputFilename, 'a') as file:
                        print('Restoring functionals: ' + restoredFnStr, file=file)
                        sys.stdout = console
                        file.close()
                    functionsToUseArray = np.logical_or(functionsToUseArray, restorableFnsArray)
                    modelActiveLib = functionsToUseArray
                    modelCoefs[restorableFnsArray] = 100  # placeholder coefficient.
                    protectedFunctionsArray[restorableFnsArray] = numItersProtection + 1  # the '+1'
                    # is due to an order-of-events weirdness (that might want attention).

            historyProtectedFnsArray.append(protectedFunctionsArray.copy())

            #%% 5. Culling phase:
            # Assess whether we want to cull and re-run; and if so, what variables and library
            # functions do we keep?

            whichIter += 1
            # If a cull in indicated, prepare for the next iteration of while loop.
            if iterateAgainFlag:
                whichCull = whichCull + 1

            preCullFunctionsToUseArray = functionsToUseArray  # As a record, to check for changes due
            # to cull step.

            # 5a. To keep culling somewhat balanced among variables, set some variables off-limits:
            librarySize = np.sum(functionsToUseArray, axis=1)
            imbalance = np.max(librarySize) - librarySize >= balancedCullNumber  # the index of the
            # variable with the biggest library
            inBalanceVarsArray = np.ones(functionsToUseArray.shape, dtype=bool)
            inBalanceVarsArray[imbalance, :] = False  # False means this var has too few functionals,
            # so it is off-limits for culling. Entire rows are uniformly True or False.

            #%% 5b. In-span culling:
            # Before the usual cull, see if any functionals are in the span of the other active
            # functionals (leave-one-out test of linear dependence). If yes, remove one of them then
            # skip the usual culling step.
            skipCullByCoeffSizeFlag = False  # default
            inSpanCullStr = ''
            skipCullByCoeffSizeDueToInSpanCullFlag = False  # reset for this iteration
            # A dict of params:
            params = dict()
            params['functionList'] = functionList 
            params['variableNames'] = variableNames  
            params['outputFilename'] = outputFilename  
            params['plottingThreshold'] = 2  # > 1 disables plotsof linearly dependent functional fits  
            params['windowLengthForFomEnvelopes'] = windowLengthForFomEnvelopes 
            params['dt'] = dt  
            params['margin'] = margin  
            params['maxFnValRatio'] =  maxFnValRatio 
            params['minNumStartIndsToUse'] = minNumStartIndsToUse
            if cullUsingInSpanFlag and not restoreFnFlag:  # if a functional was restored, skip culling
                # 5b(i). Leave-one-out linear dependence.
                includeConstantFlag = False
                LNoisy = LNoisyAll[traj]
                rSqValsLoo = \
                    findSpansOfFunctionalsLeaveOneOut_fn(functionsToUseArray,
                                                        L[fomTimepointInds, :].copy(),
                                                        LNoisy[fomTimepointInds, :].copy(), 
                                                        pointWeightsForFom, includeConstantFlag,
                                                        params)[0]  # Use 1st argout only
                    # Note: to avoid plotting candidates, set argin 6 > 1, eg 1.1

                # 5b(ii). Optionally print R-squared values:
                if printDiagnosticOfInSpanCullFlag and max(rSqValsLoo.flatten()) > 0.9:
                    console = sys.stdout
                    with open(outputFilename, 'a') as file:
                        print('Rsq leave-one-out values: \n' + str(np.round(rSqValsLoo,3)), file=file)
                        sys.stdout = console
                        file.close()

                # 5b(iii). Create a weighted coefficient array:
                weightArray = \
                    calcWeightArray_fn(postRegressionCoeffs, functionsToUseArray,
                                    imputedSizeOfFunctionals, coeffWeightsCutoffFactor, 
                                    percentileOfImputedValuesForWeights)
                wtedCoeffs = weightArray * modelCoefs

                # 5b(iv). If in-span candidates exist, cull some based on smallest weighted coef. Each
                # variable is treated separately.
                for i in range(numVars):
                    inSpanCandidates = np.logical_and(rSqValsLoo[i, :] > inSpanCullThreshold,
                                                      rSqValsLoo[i, :] < 1)
                    if enforceBoundednessFlag:
                        # Exempt what the boundedness constraint had to restore. Without this the
                        # two mechanisms fight: on collinear data a self-term is genuinely in-span,
                        # so the cull removes it, the constraint puts it back, and the run makes no
                        # progress. Boundedness wins because it is a hard requirement, and the model
                        # is integrated to score every candidate.
                        inSpanCandidates = np.logical_and(
                            inSpanCandidates, np.logical_not(boundednessRequiredFnsArray[i, :]))
                    candidateInds = np.where(inSpanCandidates)[0]
                    # The '< 1' exempts the constant functional if we set LinearRegression to
                    # automatically include a bias term (if not, then the '< 1' has no effect).
                    # Two conditions are needed to proceed:
                    if len(candidateInds) > 0 and inBalanceVarsArray[i, 0]:
                        # Calculate a max threshold. Don't cull a functional with high weighted coeff:
                        wtedCoeffsThisVar = wtedCoeffs[i, :]
                        wtedCoeffsThisVar = wtedCoeffsThisVar[np.abs(wtedCoeffsThisVar) > \
                                                            almostZeroRoundDownThreshold]
                        maxAllowedWt = np.percentile(np.abs(wtedCoeffsThisVar),
                                                    percentileThresholdForInSpanCulling)
                        # Maybe cull the one with the lowest weighted coefficient:
                        candidateWts = np.abs(wtedCoeffs[i, candidateInds])   # shorter vector
                        cullInd = candidateInds[candidateWts == min(candidateWts)][0]  # The [0] is
                        # due to the RHS being an array otherwise.
                        if np.abs(wtedCoeffs[i, cullInd]) < maxAllowedWt: # Cull if wted coeff is small
                            inSpanCullStr = inSpanCullStr + variableNames[i] + ': ' + \
                                functionList[cullInd] + '. '
                            functionsToUseArray[i, cullInd] = False
                            skipCullByCoeffSizeDueToInSpanCullFlag = True

            modelActiveLib = functionsToUseArray
            modelCoefs = modelCoefs * functionsToUseArray

            # NOTE: the boundedness constraint is deliberately NOT re-imposed here. After an in-span
            # cull the toolkit skips the FoM evolutions entirely (see 'doNotEvolveAfterInSpanCullFlag'
            # and the branch around the 'Skip FoM evolutions' message), so this model is never
            # integrated and its boundedness does not matter. Re-imposing it here also caused a
            # livelock: the constraint restored a culled self-term, the in-span cull removed it again
            # on the next iteration, and since this branch leaves 'iterateAgainFlag' untouched the
            # numCullsWithNoChange stall detector never fired.

            # We're done culling in-span functionals. Continue with 'usual' cull based on low coeffs.

            #%% 5c. Culling based on lowest coefficient (standard type of sequential threshold cull).
            # Cull library functions with low weighted coeffs. This generates a modelActiveLib for the
            # next run. The main work is done in 'cullAndAssessWhetherToRerun_fn'.
            if whichCull == 0:  # edge case
                liveVarInds = np.sum(functionsToUseArray,axis=1) > 0
                functionsToUseArray[liveVarInds, :] = True

            if restoreFnFlag:  # Case: Fill in some book-keeping values, and skip the cull
                iterateAgainFlag = True
                outputStr = 'Just restored a functional, so skip coeff-based cull this iteration.'
                minNonZeroWeightedCoeff = -1
                cullableFunctionFlag = False
                restorableFnsArray = np.zeros(functionsToUseArray.shape,dtype=bool)
            else:  # Case: Do a cull. First assign some book-keeping values
                if skipCullByCoeffSizeDueToInSpanCullFlag:  # Case: we already culled in-span fnals
                    minNonZeroWeightedCoeff = -1
                    cullableFunctionFlag = False
                    outputStr ='Removed in-span functional(s) with Rsq > ' + str(inSpanCullThreshold) + \
                        ', ' + inSpanCullStr + \
                            ' Skip coeff-based cull this iteration.'
                else:  # Case: we actually do this cull step
                    cullable = protectedFunctionsArray == 0 # any {var, fn} with > 0 entry is protected

                    # 'boundednessRequiredFnsArray' is all False unless enforceBoundednessFlag, so
                    # passing it unconditionally leaves unconstrained runs bit-exact. When the flag
                    # is on it exempts what the constraint restored, for the same reason as at the
                    # in-span cull above: otherwise this cull removes the self-term, the constraint
                    # puts it back, and the two fight to the iteration cap without progress.
                    iterateAgainFlag, functionsToUseArray, restorableFnsArray, outputStr, \
                        minNonZeroWeightedCoeff, cullableFunctionFlag = \
                            cullAndAssessWhetherToRerun_fn(modelCoefs.copy(), variableNames,
                                                        functionList, imputedSizeOfFunctionals,
                                                        cullingRulesDict, functionsToUseArray,
                                                        cullable, inBalanceVarsArray,
                                                        coeffWeightsCutoffFactor,
                                                        percentileOfImputedValuesForWeights,
                                                        boundednessRequiredFnsArray)

                    # Update the model's boolean active library array, and the coefficient array:
                    modelActiveLib = functionsToUseArray
                    modelCoefs = modelCoefs * functionsToUseArray

                    # As after the in-span cull, zeroing coeffs can break boundedness, so re-impose
                    # it on the model that the next iteration will evolve:
                    if enforceBoundednessFlag:
                        preStabilizationLib = functionsToUseArray.copy()
                        modelCoefs, functionsToUseArray, boundednessAction, numEigCalls = \
                            enforceBoundedness_fn(modelCoefs, functionsToUseArray, designMatrices)
                        boundednessActionHistory.append((whichIter, 'postCoeffCull',
                                                        boundednessAction))
                        numBoundednessEigCalls += numEigCalls
                        # This is the call site that actually restores self-terms in practice: the
                        # cull just above zeroes a diagonal entry, which makes step 3's projection
                        # infeasible on that support (sym(A) <= gamma*I < 0 forces A[v, v] < 0, see
                        # 'projectOntoBoundedAndSupport_fn'), so the restore branch fires every time.
                        # Recording it here is what lets the next iteration's cull leave it alone:
                        boundednessRequiredFnsArray = np.logical_or(
                            boundednessRequiredFnsArray,
                            np.logical_and(functionsToUseArray,
                                           np.logical_not(preStabilizationLib)))
                        modelActiveLib = functionsToUseArray
                        modelCoefs = modelCoefs * functionsToUseArray

                print('iter ' + str(whichIter) + '. ' + outputStr)
            # To track culls of variables (all functionals except maybe constant removed) and entire
            # functionals (ie the functional culled for all variables):
            removedVariableInds = np.where(np.sum(functionsToUseArray, axis = 1) == 0)[0]  # indices
            # of vars that have all zeros in their function libraries

            #%% 5d. Handling removed variables:
            # If all functionals of a variable have just been culled, the variable is now presumed
            # constant, so all functionals for which that variable is an argin must be culled. This
            # changes the picture such that we don't trust our cull history, so we start over.
            # So if we have just removed a variable, we want to start over with functionsToUseArray
            # restored such that rows == True for all surviving variables, except for functionals using
            # the removed variable as an argin, which are all set to False.
            # If we would stop anyway (ie stopIfVarIsCulledFlag == True), then don't run this section.
            if len(removedVariableInds) > numRemovedVariables and not stopIfVarIsCulledFlag:
                numCullsWithNoChange = 0
                restoredFnArray = np.ones(functionsToUseArray.shape, dtype=bool)
                restoredFnArray[removedVariableInds, :] = False
                # 2. Now zero out all functions that involve zeroed-out variables, ie zero out columns
                # that contain the variable name:
                for i in removedVariableInds:
                    varName = variableNames[i]
                    for j in range(restoredFnArray.shape[1]):
                        if varName in functionList[j]: # exception for the constant function
                            restoredFnArray[:, j] = False
                functionsToUseArray = restoredFnArray
                modelCoefs = functionsToUseArray.astype(float)
                print('Note: Removed a new variable, re-starting the cull steps.')
                console = sys.stdout
                with open(outputFilename, 'a') as file:
                    print('Note: Removed a new variable, re-starting the cull steps.', file=file)
                    sys.stdout = console
                    file.close()
                whichCull = -1  # It gets incremented before leaving the 'cull' block.

            # 5e. Assorted book-keeping and printouts:
            numRemovedVariables = len(removedVariableInds)
            removedVariableStr =  str(np.array(variableNames)[removedVariableInds])  # Each printed message
            # lists all variables removed to date.
            # Track functionals that have been removed in all variables, for fun only:
            removedFunctionInds = np.where(np.sum(functionsToUseArray, axis = 0) == 0)[0]
            # See which of these are new:
            newRemovedFnInds = removedFunctionInds.copy()
            for i in range(len(newRemovedFnInds)):
                if removedFunctionInds[i] in culledFunctionIndices:
                    newRemovedFnInds[i] = -1
            newRemovedFnInds = newRemovedFnInds[np.where(newRemovedFnInds > -1)[0]]
            newRemovedFunctionStr = str(np.array(functionList)[newRemovedFnInds])

            # Print to console (Optional diagnostic):
            removedVariableStr = removedVariableStr.replace('[','')
            removedVariableStr = removedVariableStr.replace(']','')
            if len(removedVariableStr) > 0:
                variablesRemovedPrintStr = 'Removed variables: ' +  removedVariableStr + '. '
            else:
                variablesRemovedPrintStr = ''
            newRemovedFunctionStr = newRemovedFunctionStr.replace('[','')
            newRemovedFunctionStr = newRemovedFunctionStr.replace(']','')
            if len(newRemovedFunctionStr) > 0:
                functionsRemovedPrintStr = ' Newly-removed functional column(s): ' + \
                    newRemovedFunctionStr + '. '
            else:
                functionsRemovedPrintStr = ''

            # Print results of culling:
            console = sys.stdout
            with open(outputFilename, 'a') as file:
                print('\n' + 'Cull ' + str(whichCull) + ' results: ' + \
                variablesRemovedPrintStr + outputStr, file=file)  # + functionsRemovedPrintStr)
                sys.stdout = console
                file.close()

            # update culledFunctionIndices:
            culledFunctionIndices = removedFunctionInds.copy()

            # Culls are now done and changes are recorded.

            # 5f. Update some flags, to decide whether to iterate again:
            if iterateAgainFlag == False:
                numCullsWithNoChange += 1
            else:
                numCullsWithNoChange = 0   # if this cull had some effect, reset the counter.

            # Modify iterateAgainFlag according to other constraints:
            if whichCull > maxNumCulls:
                iterateAgainFlag = False
            if np.sum(functionsToUseArray.flatten()) == 0:
                iterateAgainFlag = False

            # Stop this trajectory if the iteration cap is reached:
            if maxNumIterations is not None and whichIter >= maxNumIterations:
                iterateAgainFlag = False

            # Stop this trajectory if its own wall-clock budget is exceeded (every trajectory gets
            # the same budget, timed from when its regress-cull loop started):
            if maxRunTimeSecs is not None and (time.time() - trajStartTime) > maxRunTimeSecs:
                iterateAgainFlag = False

            # If weighted coeffs are stable and above the max cutoff threshold, and we have not just
            # restored a functional, stop the iterations:
            noCoeffsCutFlag = len(newRemovedFunctionStr) == 0 and ('Also culled' not in outputStr)

            # Maybe stop the run due to a variable being wiped out:
            if stopIfVarIsCulledFlag and len(removedVariableInds) > 0:
                iterateAgainFlag = False
        # End of while iterateAgainFlag loop (started around line 2340). When we exit this loop, we are
        # done with the regress-cull iterations.

        #%% 6. Now save this trajectory's history lists:
        historyCoeffArrayAll[traj] = historyCoeffArray
        boundednessActionHistoryAll[traj] = boundednessActionHistory
        numBoundednessEigCallsAll[traj] = numBoundednessEigCalls
        historyProtectedFnsArrayAll[traj] = historyProtectedFnsArray
        historyWhichIterAll[traj] = historyWhichIter
        historyWhichCullAll[traj] = historyWhichCull
        historyInBoundsFoMAll[traj] = historyInBoundsFoM
        historyInEnvelopeFoMAll[traj] = historyInEnvelopeFoM
        historyStdDevFoMAll[traj] = historyStdDevFoM
        historyMeanFoMAll[traj] = historyMeanFoM
        historyMedianFoMAll[traj] = historyMedianFoM
        historyFftCorrelationFoMAll[traj] = historyFftCorrelationFoM
        historyFftPowerAll[traj] = historyFftPower
        historyHistogramCorrelationFoMAll[traj] = historyHistogramCorrelationFoM
        historyHistogramsAll[traj] = historyHistograms
        historyHistogramBinsAll[traj] = historyHistogramBins
        historyMinHistCorrelationForEvolutionsAll[traj] = historyMinHistCorrelationForEvolutions
        historyXDotDiffEnvelopeAll[traj] = historyXDotDiffEnvelope
        historyXDotInBoundsFoMAll[traj] = historyXDotInBoundsFoM
        historyXDotHistogramCorrelationFoMAll[traj] = historyXDotHistogramCorrelationFoM
        historyXDotHistogramsAll[traj] = historyXDotHistograms
        historyXDotHistogramBinsAll[traj] = historyXDotHistogramBins

        # for xVal:
        historyCoeffArrayValAll[traj] = historyCoeffArrayVal  # to save coeff matrices of each step.
        historyProtectedFnsArrayValAll[traj] = historyProtectedFnsArrayVal
        historyWhichIterValAll[traj] = historyWhichIterVal
        historyWhichCullValAll[traj] = historyWhichCullVal
        historyInBoundsFoMValAll[traj] = historyInBoundsFoMVal
        historyInEnvelopeFoMValAll[traj] = historyInEnvelopeFoMVal
        historyStdDevFoMValAll[traj] = historyStdDevFoMVal
        historyMeanFoMValAll[traj] = historyMeanFoMVal
        historyMedianFoMValAll[traj] = historyMedianFoMVal
        historyFftCorrelationFoMValAll[traj] = historyFftCorrelationFoMVal
        historyFftPowerValAll[traj] = historyFftPowerVal
        historyHistogramCorrelationFoMValAll[traj]= historyHistogramCorrelationFoMVal
        historyHistogramsValAll[traj] = historyHistogramsVal
        historyHistogramBinsValAll[traj] = historyHistogramBinsVal
        historyMinHistCorrelationForEvolutionsValAll[traj] = historyMinHistCorrelationForEvolutionsVal
        historyXDotDiffEnvelopeValAll[traj] = historyXDotDiffEnvelopeVal
        historyXDotInBoundsFoMValAll[traj] = historyXDotInBoundsFoMVal
        historyXDotHistogramCorrelationFoMValAll[traj] = historyXDotHistogramCorrelationFoMVal
        historyXDotHistogramsValAll[traj] = historyXDotHistogramsVal
        historyXDotHistogramBinsValAll[traj] = historyXDotHistogramBinsVal

        # End of loop "for traj in range(numTrajTrain):"

    #%% 7. Save the results, for use by 'plotSelectedIterations.py':
    # Save two things: original system information, and histories of iteration results.

    results = dict()

    # Original system trajectories and associated parameters:
    results['numTrajTrain'] = numTrajTrain
    results['variableNames'] = variableNames
    results['functionList'] = functionList
    results['dt'] = dt
    results['marginInSecs'] = marginInSecs
    results['numSecsInTrain'] = numSecsInTrain
    results['numSecsInTest'] = numSecsInTest
    results['numTrajTest'] = numTrajTest
    results['tTrain'] = tTrain
    results['xTrainAll'] = xTrainAll
    results['xDotTrainUnsmoothedAll'] = xDotTrainUnsmoothedAll
    results['xDotTrainAll'] = xDotTrainAll
    results['indsValAll'] = indsValAll
    results['tTest'] = tTest
    results['xTestAll'] = xTestAll

    results['xTrainFftPowerAll'] = xTrainFftPowerAll
    results['xTrainHistogramAll'] = xTrainHistogramAll
    results['xTrainHistogramBinsAll'] = xTrainHistogramBinsAll
    results['xDotTrainHistogramAll'] = xDotTrainHistogramAll
    results['xDotTrainHistogramBinsAll'] = xDotTrainHistogramBinsAll

    results['simInitCondsAll'] = simInitCondsAll
    results['LAll'] = LAll
    results['recipes'] = recipes
    results['fomTimepointInds'] = fomTimepointInds
    results['derivAndFnalSmoothType'] = 'hamming'
    results['hammDerivs'] = hammDerivs
    results['LTestAll'] = LTestAll
    results['pointWeightsForFomAll'] = pointWeightsForFomAll
    results['LNoisyAll'] = LNoisyAll

    results['outputFilename'] = outputFilename
    results['windowLengthForFomEnvelopes'] = windowLengthForFomEnvelopes 
    results['maxFnValRatio'] = maxFnValRatio
    results['minNumStartIndsToUse'] = minNumStartIndsToUse

    # The other argins that used to be hard-coded, saved so a run can be reproduced from its pickle:
    results['hammingWindowLengthForData'] = hammingWindowLengthForData
    results['hammingWindowLengthForDerivsAndFnals'] = hammingWindowLengthForDerivsAndFnals
    results['regressOnFftAlpha'] = regressOnFftAlpha
    results['balancedCullNumber'] = balancedCullNumber
    results['numRegressionSegments'] = numRegressionSegments
    results['minNumSegmentResultsToUse'] = minNumSegmentResultsToUse
    results['minAllowedWeightedCoeff'] = minAllowedWeightedCoeff
    results['overlapFraction'] = overlapFraction

    # The trapping-style boundedness constraint. Note that 'enforceBoundednessFlag' is the value
    # actually used, ie already switched off if polynomialLibraryDegree != 1:
    results['enforceBoundednessFlag'] = enforceBoundednessFlag
    results['boundednessMargin'] = boundednessMargin
    results['strictBoundednessMargin'] = strictBoundednessMargin
    results['boundednessRidgeLadder'] = boundednessRidgeLadder
    results['boundednessNormType'] = boundednessNormType
    results['boundednessMethod'] = boundednessMethod
    results['boundednessNuLadder'] = boundednessNuLadder
    results['boundednessPenaltyLadder'] = boundednessPenaltyLadder
    results['boundednessPenaltySmoothing'] = boundednessPenaltySmoothing
    results['boundednessMaxNumFreeEntries'] = boundednessMaxNumFreeEntries
    results['boundednessActionHistoryAll'] = boundednessActionHistoryAll
    results['numBoundednessEigCallsAll'] = numBoundednessEigCallsAll

    # Histories of all iterations in this run:
    results['historyCoeffArrayAll'] =  historyCoeffArrayAll
    results['historyProtectedFnsArrayAll'] = historyProtectedFnsArrayAll
    results['historyWhichIterAll'] =  historyWhichIterAll   
    results['historyWhichCullAll'] =  historyWhichCullAll
    results['historyInBoundsFoMAll'] = historyInBoundsFoMAll 
    results['historyInEnvelopeFoMAll'] = historyInEnvelopeFoMAll 
    results['historyStdDevFoMAll'] = historyStdDevFoMAll 
    results['historyMeanFoMAll'] = historyMeanFoMAll 
    results['historyMedianFoMAll'] =  historyMedianFoMAll
    results['historyFftCorrelationFoMAll'] = historyFftCorrelationFoMAll 
    results['historyFftPowerAll'] = historyFftPowerAll 
    results['historyHistogramCorrelationFoMAll'] = historyHistogramCorrelationFoMAll  
    results['historyHistogramsAll'] =  historyHistogramsAll
    results['historyHistogramBinsAll'] = historyHistogramBinsAll 
    results['historyMinHistCorrelationForEvolutionsAll'] = \
        historyMinHistCorrelationForEvolutionsAll
    results['historyXDotInBoundsFoMAll'] = historyXDotInBoundsFoMAll 
    results['historyXDotDiffEnvelopeAll'] =  historyXDotDiffEnvelopeAll
    results['historyXDotHistogramCorrelationFoMAll'] = historyXDotHistogramCorrelationFoMAll
    results['historyXDotHistogramsAll'] = historyXDotHistogramsAll 
    results['historyXDotHistogramBinsAll'] = historyXDotHistogramBinsAll 

    # for xVal: 
    results['historyCoeffArrayValAll'] = historyCoeffArrayValAll 
    results['historyProtectedFnsArrayValAll'] =  historyProtectedFnsArrayValAll
    results['historyWhichIterValAll'] =  historyWhichIterValAll
    results['historyWhichCullValAll'] = historyWhichCullValAll 
    results['historyInBoundsFoMValAll'] =  historyInBoundsFoMValAll 
    results['historyInEnvelopeFoMValAll'] = historyInEnvelopeFoMValAll 
    results['historyStdDevFoMValAll'] =  historyStdDevFoMValAll
    results['historyMeanFoMValAll'] = historyMeanFoMValAll 
    results['historyMedianFoMValAll'] = historyMedianFoMValAll 
    results['historyFftCorrelationFoMValAll'] = historyFftCorrelationFoMValAll 
    results['historyFftPowerValAll'] = historyFftPowerValAll 
    results['historyHistogramCorrelationFoMValAll'] = historyHistogramCorrelationFoMValAll
    results['historyHistogramsValAll'] = historyHistogramsValAll
    results['historyHistogramBinsValAll'] =  historyHistogramBinsValAll
    results['historyMinHistCorrelationForEvolutionsValAll'] = \
        historyMinHistCorrelationForEvolutionsValAll 
    results['historyXDotDiffEnvelopeValAll'] = historyXDotDiffEnvelopeValAll
    results['historyXDotInBoundsFoMValAll'] = historyXDotInBoundsFoMValAll
    results['historyXDotHistogramCorrelationFoMValAll'] = \
        historyXDotHistogramCorrelationFoMValAll 
    results['historyXDotHistogramsValAll'] = historyXDotHistogramsValAll
    results['historyXDotHistogramBinsValAll'] = historyXDotHistogramBinsValAll 

    # Save dict in a pickle file:
    with open(pickleFilename, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
        
    '''  All fit-and-cull iterations are done. Now plot FoMs and maybe time-series: '''
    
    # Reminder of where data was saved: 
    print('')
    print('See ' + outputFilename + ' for detailed progress of run.')
    print('Load ' + pickleFilename + ' to get complete run data after completion. \n ') 

    #%% 8. Plot figures of merit mosaics:
    # These are an important tool to assess the accuracy of models as they become progressively
    # more sparse.
    # Format: For each train trajectory, plot one mosaic 5 x 4: first two cols = home trajectory
    # results, last 2 cols = results on each val trajectory, using the home trajectory model.

    if showFiguresOfMeritMosaicsFlag:

        # Pack d, the dataDict which is argin for the FoM mosaic function:
        d = dict()
        d['seed'] = seed
        d['numTrajTrain'] = numTrajTrain
        d['variableNames'] = variableNames
        d['functionList'] = functionList
        d['outputFilename'] = outputFilename

        d['indsValAll'] = indsValAll
        d['historyWhichIterAll'] = historyWhichIterAll
        d['historyCoeffArrayAll'] = historyCoeffArrayAll
        d['historyMinHistCorrelationForEvolutionsAll'] = historyMinHistCorrelationForEvolutionsAll
        d['historyInBoundsFoMAll'] = historyInBoundsFoMAll
        d['historyInEnvelopeFoMAll'] = historyInEnvelopeFoMAll
        d['historyStdDevFoMAll'] = historyStdDevFoMAll
        d['historyHistogramCorrelationFoMAll'] = historyHistogramCorrelationFoMAll
        d['historyXDotDiffEnvelopeAll'] = historyXDotDiffEnvelopeAll
        d['historyMinHistCorrelationForEvolutionsValAll'] = \
            historyMinHistCorrelationForEvolutionsValAll
        d['historyInBoundsFoMValAll'] = historyInBoundsFoMValAll
        d['historyInEnvelopeFoMValAll'] = historyInEnvelopeFoMValAll
        d['historyFftCorrelationFoMAll'] = historyFftCorrelationFoMAll
        d['historyXDotHistogramCorrelationFoMAll'] = historyXDotHistogramCorrelationFoMAll
        d['historyStdDevFoMValAll'] = historyStdDevFoMValAll
        d['historyHistogramCorrelationFoMValAll'] = historyHistogramCorrelationFoMValAll
        d['historyFftCorrelationFoMValAll'] = historyFftCorrelationFoMValAll
        d['historyXDotHistogramCorrelationFoMValAll'] = historyXDotHistogramCorrelationFoMValAll
        d['historyXDotDiffEnvelopeValAll'] = historyXDotDiffEnvelopeValAll

        # Call the mosaic function:
        plotFiguresOfMeritMosaics_fn(d)

    return 1
