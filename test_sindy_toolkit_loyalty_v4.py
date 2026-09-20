#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINDy toolkit на всех пяти траекториях v4 разом, с ограничением ограниченности (trapping SINDy).

Пять групп одного сегмента (`load_loyalty_v4_data.to_trajectory`) передаются как
пять обучающих траекторий в одном вызове `sindy_toolkit_func` — многотраекторный
режим тулкита, а не пять отдельных прогонов. Отбор признаков повторяет
`thesis_demo.ipynb` (полное 86-мерное пространство, без SVD-сжатия):
MIN_MAX_SHARE=0.03 считается по всем пяти группам сразу, плюс явный дроп
['x50', 'x5']; 104 -> 86 признаков.

`maxRunTimeSecs=600` ограничивает regress-cull цикл каждой из пяти траекторий
десятью минутами независимо друг от друга (см. sindy_toolkit_function.py).

`sindy_toolkit_func` вызывается с `enforceBoundednessFlag=True,
boundednessMethod='trapping'`: без ограничения часть итераций даёт модели с
неограниченными траекториями, и figures of merit на них бессмысленны, потому что
тулкит интегрирует каждого кандидата. Ограничение реализовано только для
`polynomialLibraryDegree=1`, что здесь и используется.

Про gamma: признаки v4 — доли, и в библиотеке степени 1 активен константный
функционал, поэтому `boundednessMargin=0` автоматически ужесточается до
`strictBoundednessMargin` — при аффинном члене маргинальной ограниченности
недостаточно, см. `stabilizeLinearModel_fn`.

Запуск: python test_sindy_toolkit_loyalty_v4.py
Результат — `runResults_loyaltyV4_initialRun_<seed>.pickle`.
"""

import glob
import logging
import os
import pickle
import warnings

import numpy as np

from load_loyalty_v4_data import load_loyalty_df, to_trajectory
from sindy_toolkit_function import sindy_toolkit_func
from toolkitSupportFunctions import boundednessViolation_fn

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Настройки прогона -------------------------------------------------------

MIN_MAX_SHARE = 0.03
DROP_FEATURES = ['x50', 'x5', 'x62']  # как в thesis_demo.ipynb
TRAIN_RANGE = 650           # из 731 даты; остальное - тест
MAX_RUN_TIME_SECS = 600     # на каждую из пяти траекторий
DATASET_NAME = 'loyaltyV4_run1'
NORM_TYPE = 'spectral'      # точное условие для линейной модели; 'identity' - буквальное условие статей
BOUNDEDNESS_MARGIN = 0.

# --- Данные ------------------------------------------------------------------

df, all_features = load_loyalty_df()

# Порог берётся по всем группам сразу
feature_cols = [
    c for c in all_features
    if c not in DROP_FEATURES and df[c].max() >= MIN_MAX_SHARE
]

_, train_data_full, group_ids = to_trajectory(df, feature_cols)
numVars = train_data_full.shape[2]

train_noisy = train_data_full[:, :TRAIN_RANGE, :]
test_noisy = train_data_full[:, TRAIN_RANGE:, :]
variableNames = feature_cols

logger.info(f"группы:    {group_ids}")
logger.info(f"признаков: {len(all_features)} -> {numVars}")
logger.info(f"train_data={train_noisy.shape} test_data={test_noisy.shape}")
logger.info(f"maxRunTimeSecs={MAX_RUN_TIME_SECS} на каждую из {len(group_ids)} траекторий")

for stale in glob.glob('runResults_' + DATASET_NAME + '_*.pickle'):
    os.remove(stale)
for stale in glob.glob('outputFile_' + DATASET_NAME + '_*'):
    os.remove(stale)

sindy_toolkit_func(
    train_data=train_noisy,
    test_data=test_noisy,
    datasetName=DATASET_NAME,
    variableNames=variableNames,
    dt=1,
    marginInSecs=5,
    polynomialLibraryDegree=1,
    useFullLibraryFlag=True,
    minNumStartIndsToUse=100,
    hammingWindowLengthForData=31,
    hammingWindowLengthForDerivsAndFnals=10,
    maxFnValRatio=50,
    regressOnFftAlpha=0.1,
    balancedCullNumber=20,
    numRegressionSegments=3,
    minNumSegmentResultsToUse=2,
    minAllowedWeightedCoeff=0.001,
    overlapFraction=0,
    maxRunTimeSecs=MAX_RUN_TIME_SECS,
    enforceBoundednessFlag=True,
    boundednessMargin=BOUNDEDNESS_MARGIN,
    boundednessNormType=NORM_TYPE,
    boundednessMethod='trapping',
)

written = glob.glob('runResults_' + DATASET_NAME + '_*.pickle')
if len(written) != 1:
    raise RuntimeError('ожидался ровно один пикл для ' + DATASET_NAME + ', есть ' + str(written))
with open(written[0], 'rb') as f:
    results = pickle.load(f)


def violationsPerIteration(results, normType):
    """Нарушение ограниченности линейного блока на каждой итерации каждой траектории."""
    nVars = len(results['variableNames'])
    out = []
    for historyCoeffArray in results['historyCoeffArrayAll']:
        for coeffArray in historyCoeffArray:
            out.append(boundednessViolation_fn(coeffArray[:, 1:1 + nVars], normType))

    return np.array(out)


violations = violationsPerIteration(results, NORM_TYPE)

tol = 1e-9
print('\n' + '=' * 78)
print('признаков: %d, обучение: %d точек, норма: %s, gamma: %g'
      % (numVars, TRAIN_RANGE, NORM_TYPE, BOUNDEDNESS_MARGIN))
print('пикл: %s' % written[0])
print('итераций записано: %d' % len(violations))
print('худшее нарушение (<= %g — модель ограничена): %+.4e   (%d из %d итераций неограничены)'
      % (BOUNDEDNESS_MARGIN, violations.max(),
         int(np.sum(violations > BOUNDEDNESS_MARGIN + tol)), len(violations)))

counts = {}
for perTraj in results['boundednessActionHistoryAll']:
    for _whichIter, _site, action in perTraj:
        counts[action] = counts.get(action, 0) + 1
print('что делало ограничение: ' + (str(counts) if counts else '(ни разу не сработало)'))
print('разложений на собственные числа: %s' % results['numBoundednessEigCallsAll'])

allBounded = bool(violations.max() <= BOUNDEDNESS_MARGIN + tol)
print('\n' + ('PASS: все итерации ограничены' if allBounded else
              'FAIL: есть неограниченная итерация с включённым ограничением'))
