#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скалярная метрика качества итерации для SINDy-тулкита Delahunt-Kutz — замена ручного просмотра
мозаик FoM ('plotFiguresOfMeritMosaics.py').

Мотивация: мозаика содержит 11 подграфиков, на каждом numVariables серий точек по всем итерациям
культинга. На синтетике (3 переменные, 40 итераций) её можно прочитать глазами; на лояльности
(87 переменных, 1164 итерации) — нет. Метрика сворачивает те же самые FoM в одно число на итерацию
и применяет то правило отбора, которое человек применяет к мозаике на глаз: взять плато лучшего
качества и выбрать в нём самую разреженную модель.

Как считается (подробности — в docstring 'computeIterationScores_fn'):

  1. Каждый FoM переводится в качество q = 1 / (1 + |value - ideal| / tol) на переменную.
     'tol' — это отклонение, при котором q = 0.5, и берётся оно не с потолка, а из
     'fomChangesDict' самого тулкита (блок USER ENTRIES в 'sindy_toolkit_function.py'): там
     заданы уровни, ниже которых FoM считается «приличным». Поэтому q = 0.5 ровно на границе
     приличия по меркам тулкита, q > 0.5 — лучше неё.
  2. Свёртка по переменным: sqrt(mean * p10). Высокий балл требует, чтобы и типичная, и худшая
     десятая часть переменных были хороши. Насыщение на шаге 1 не даёт одной разошедшейся
     переменной обнулить балл всей итерации.
  3. Свёртка по FoM: взвешенное среднее геометрическое. Провал по любой оси тянет итоговый балл
     вниз — арифметическое среднее позволило бы отличному in-bounds замаскировать нулевой FFT.
  4. Блоки траектории (70%) и производных (30%): в режиме высокого шума поведение проэволюционной
     траектории информативнее невязки регрессии — это исходная идея метода.

Итог: 'score' в (0, 1], сравнимый между итерациями и между запусками, с абсолютной трактовкой
порога 0.5. Выбор итерации — 'selectBestIteration_fn'.

Использование:
    python iterationScore.py runResults_loyaltyChknone_initialRun_681.pickle
или
    from iterationScore import selectBestIteration_fn, printIterationReport_fn
    sel = selectBestIteration_fn(results)          # results — распакованный pickle или сам dict
    printIterationReport_fn(sel, results)
"""

import pickle
import sys

import numpy as np

# Спецификация FoM. 'tol' — отклонение от идеала, при котором качество = 0.5.
# Источник значений 'tol' — 'fomChangesDict' в 'sindy_toolkit_function.py': там для каждого FoM
# задан минимальный «приличный» уровень предыдущей итерации, и tol = |ideal - этот уровень|.
# Для xDot-FoM в тулките таких уровней нет, поэтому взяты аналогичные по смыслу (помечено).
FOM_SPECS = (
    # (ключ в results, ключ Val-истории, идеал, tol, блок, вес внутри блока)
    ('historyInBoundsFoMAll', 'historyInBoundsFoMValAll', 1.0, 0.20, 'traj', 1.0),
    ('historyInEnvelopeFoMAll', 'historyInEnvelopeFoMValAll', 1.0, 0.40, 'traj', 1.0),
    ('historyStdDevFoMAll', 'historyStdDevFoMValAll', 0.0, 0.20, 'traj', 1.0),
    ('historyHistogramCorrelationFoMAll', 'historyHistogramCorrelationFoMValAll',
     1.0, 0.40, 'traj', 1.0),
    ('historyFftCorrelationFoMAll', 'historyFftCorrelationFoMValAll', 1.0, 0.30, 'traj', 1.0),
    ('historyMinHistCorrelationForEvolutionsAll', 'historyMinHistCorrelationForEvolutionsValAll',
     1.0, 0.02, 'traj', 1.0),
    ('historyXDotDiffEnvelopeAll', 'historyXDotDiffEnvelopeValAll',
     0.0, 0.30, 'deriv', 1.0),  # tol подобран по смыслу: 80-й перцентиль ошибки в 30% от max|xDot|
    ('historyXDotHistogramCorrelationFoMAll', 'historyXDotHistogramCorrelationFoMValAll',
     1.0, 0.40, 'deriv', 1.0),  # tol по аналогии с гистограммной корреляцией траектории
)

SHORT_NAMES = {
    'historyInBoundsFoMAll': 'inBounds',
    'historyInEnvelopeFoMAll': 'inEnvelope',
    'historyStdDevFoMAll': 'stdDev',
    'historyHistogramCorrelationFoMAll': 'histCorr',
    'historyFftCorrelationFoMAll': 'fftCorr',
    'historyMinHistCorrelationForEvolutionsAll': 'evolStab',
    'historyXDotDiffEnvelopeAll': 'xDotErr',
    'historyXDotHistogramCorrelationFoMAll': 'xDotHist',
}

trajBlockWeight = 0.7  #!!!!!# Вес блока траектории против блока производных (0.3).
coeffIsActiveThreshold = 1e-5  # так же, как в 'plotFiguresOfMeritMosaics.py'.


def _as2d_fn(a):
    """История FoM -> массив numIters x numVars. Строки-заглушки имеют лишнюю ось (см. 'dummyVal'
    в 'sindy_toolkit_function.py'), 'plotFoMSubplot_fn' раздавливает её так же."""
    a = np.array(a, dtype=float)
    if a.ndim == 3:
        a = a[:, 0, :]
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _aggregateOverVars_fn(q):
    """Свёртка качества по переменным: sqrt(mean * p10) — среднее геометрическое типичного и
    худше-децильного значения. Требует, чтобы и большинство переменных, и хвост были в порядке."""
    with np.errstate(invalid='ignore'):
        if np.all(np.isnan(q)):
            return np.nan
        mean = np.nanmean(q)
        p10 = np.nanpercentile(q, 10)
    return float(np.sqrt(max(mean, 0.0) * max(p10, 0.0)))


def _rollingMedian_fn(x, window):
    """Скользящая медиана по индексу строки, с урезанным окном на краях. NaN игнорируются."""
    if window <= 1:
        return x.copy()
    half = window // 2
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        chunk = x[lo:hi]
        if np.any(np.isfinite(chunk)):
            out[i] = np.nanmedian(chunk)
    return out


def computeIterationScores_fn(d, traj=0, useValIfAvailable=True):
    """
    Посчитать скалярный балл каждой итерации культинга для одной обучающей траектории.

    Балл каждого FoM: q = 1 / (1 + |value - ideal| / tol), покомпонентно по переменным.
    Функция насыщающая, поэтому разошедшаяся переменная (у 'xDotDiffEnvelope' встречаются
    значения ~1e4) даёт q -> 0, но не ломает шкалу и не обнуляет всю итерацию, если остальные
    переменные в порядке. Дальше — свёртка по переменным ('_aggregateOverVars_fn'), затем
    взвешенное среднее геометрическое по FoM внутри блока, затем блоки:

        score = scoreTraj ** 0.7 * scoreDeriv ** 0.3

    Вырожденные FoM отбрасываются автоматически: если история FoM константна по всем итерациям
    (так ведёт себя 'minHistCorrelationForEvolutions' при numEvolutionsForTrainFom == 1, он всюду
    равен 1), он не несёт информации о выборе итерации и исключается из свёртки.

    Валидационные траектории: если Val-истории непусты, для каждой итерации берётся худшая
    валидационная траектория, и итоговый балл = scoreTrain ** 0.4 * scoreVal ** 0.6 — обобщение
    важнее посадки на обучающую траекторию.

    Parameters
    ----------
    d : dict. Результаты запуска ('runResults_*.pickle' или возвращаемый 'results' dict).
    traj : int. Номер обучающей траектории.
    useValIfAvailable : bool. Учитывать ли валидационные FoM, когда они есть.

    Returns
    -------
    dict со следующими массивами длины numIters (numIters — число записанных итераций):
        'whichIter'   : номера итераций (то, что идёт по оси x в мозаике),
        'score'       : итоговый балл в (0, 1],
        'scoreTrain', 'scoreVal', 'scoreTraj', 'scoreDeriv',
        'components'  : dict короткое имя FoM -> балл по этому FoM,
        'nActiveTotal': число активных функционалов во всей модели,
        'nActiveMax'  : максимум активных функционалов на одну переменную,
        'isStale'     : True, если FoM скопированы с предыдущей итерации (эволюции пропускались),
        'isDummy'     : True, если FoM — заглушка (-1),
        'isValid'     : пригодна ли итерация для выбора,
        'droppedFoms' : список FoM, исключённых как константные,
        'perVarQuality': numIters x numVars, качество на переменную (для диагностики).
    """
    coef = np.array(d['historyCoeffArrayAll'][traj], dtype=float)  # numIters x numVars x numFnals
    active = np.abs(coef) > coeffIsActiveThreshold
    nActivePerVar = np.sum(active, axis=2)
    numIters, numVars = nActivePerVar.shape

    whichIter = np.array(d['historyWhichIterAll'][traj], dtype=int)[:numIters]

    components = {}
    droppedFoms = []
    qualityStack = {'traj': [], 'deriv': []}
    weightStack = {'traj': [], 'deriv': []}
    perVarStack = []

    isDummy = np.zeros(numIters, dtype=bool)

    for key, valKey, ideal, tol, block, weight in FOM_SPECS:
        train = _as2d_fn(d[key][traj])[:numIters]
        name = SHORT_NAMES[key]

        # Строки-заглушки: тулкит пишет -1 по всем переменным, когда эволюции не считались вовсе.
        isDummy |= np.all(train == -1, axis=1)

        # Константный FoM ничего не говорит о выборе итерации — исключаем.
        finite = train[np.isfinite(train)]
        if finite.size == 0 or np.allclose(finite, finite.flat[0]):
            droppedFoms.append(name)
            continue

        qTrain = 1.0 / (1.0 + np.abs(train - ideal) / tol)

        qVal = None
        if useValIfAvailable:
            valHist = np.array(d[valKey][traj], dtype=float)
            # numIters x numValTraj x numVars, либо пусто при единственной обучающей траектории.
            if valHist.ndim == 3 and valHist.size > 0:
                nv = min(len(valHist), numIters)
                qv = 1.0 / (1.0 + np.abs(valHist[:nv] - ideal) / tol)
                qVal = np.full((numIters, numVars), np.nan)
                # Худшая валидационная траектория на каждой итерации и переменной.
                qVal[:nv] = np.nanmin(qv, axis=1)

        scoreTrainFom = np.array([_aggregateOverVars_fn(row) for row in qTrain])
        components[name] = scoreTrainFom
        qualityStack[block].append(scoreTrainFom)
        weightStack[block].append(weight)
        perVarStack.append(qTrain)

        if qVal is not None:
            valName = name + 'Val'
            scoreValFom = np.array([_aggregateOverVars_fn(row) for row in qVal])
            components[valName] = scoreValFom
            qualityStack.setdefault('trajVal' if block == 'traj' else 'derivVal', []).append(
                scoreValFom)
            weightStack.setdefault('trajVal' if block == 'traj' else 'derivVal', []).append(weight)

    def _geoMean(block):
        vals = qualityStack.get(block, [])
        if not vals:
            return np.full(numIters, np.nan)
        w = np.array(weightStack[block], dtype=float)
        w = w / w.sum()
        stack = np.clip(np.array(vals), 1e-12, None)  # log(0) не нужен, качество строго > 0
        return np.exp(np.sum(w[:, None] * np.log(stack), axis=0))

    scoreTraj = _geoMean('traj')
    scoreDeriv = _geoMean('deriv')
    scoreTrain = _combineBlocks_fn(scoreTraj, scoreDeriv)

    scoreTrajVal = _geoMean('trajVal')
    scoreDerivVal = _geoMean('derivVal')
    scoreVal = _combineBlocks_fn(scoreTrajVal, scoreDerivVal)

    if np.all(np.isnan(scoreVal)):
        score = scoreTrain
    else:
        score = np.exp(0.4 * np.log(np.clip(scoreTrain, 1e-12, None)) +
                       0.6 * np.log(np.clip(scoreVal, 1e-12, None)))

    # Несвежие FoM: тулкит пропускает эволюции после культинга in-span функционала и при слишком
    # большой библиотеке, и тогда все FoM повторяются с прошлой итерации. Такая строка утверждает,
    # что более разреженная модель ровно так же хороша, — при выборе разреженного максимума это
    # ровно та ошибка, которую нельзя допустить, поэтому такие итерации исключаются из отбора.
    fomRows = np.concatenate(
        [_as2d_fn(d[key][traj])[:numIters] for key, *_ in FOM_SPECS], axis=1)
    isStale = np.zeros(numIters, dtype=bool)
    with np.errstate(invalid='ignore'):
        isStale[1:] = np.all((fomRows[1:] == fomRows[:-1]) |
                             (np.isnan(fomRows[1:]) & np.isnan(fomRows[:-1])), axis=1)

    isValid = np.isfinite(score) & ~isStale & ~isDummy & (nActivePerVar.min(axis=1) > 0)

    return {
        'whichIter': whichIter,
        'score': score,
        'scoreTrain': scoreTrain,
        'scoreVal': scoreVal,
        'scoreTraj': scoreTraj,
        'scoreDeriv': scoreDeriv,
        'components': components,
        'nActiveTotal': nActivePerVar.sum(axis=1),
        'nActiveMax': nActivePerVar.max(axis=1),
        'nActivePerVar': nActivePerVar,
        'isStale': isStale,
        'isDummy': isDummy,
        'isValid': isValid,
        'droppedFoms': droppedFoms,
        'perVarQuality': (np.nanmean(np.array(perVarStack), axis=0) if perVarStack
                          else np.full((numIters, numVars), np.nan)),
        'numVars': numVars,
        'numFunctionals': coef.shape[2],
    }


def _combineBlocks_fn(scoreTraj, scoreDeriv):
    """Блок траектории с весом 0.7, блок производных с 0.3. Если один блок отсутствует целиком
    (все его FoM отброшены как константные), берётся второй."""
    tOk = np.isfinite(scoreTraj)
    dOk = np.isfinite(scoreDeriv)
    out = np.full(len(scoreTraj), np.nan)
    both = tOk & dOk
    out[both] = (np.power(np.clip(scoreTraj[both], 1e-12, None), trajBlockWeight) *
                 np.power(np.clip(scoreDeriv[both], 1e-12, None), 1 - trajBlockWeight))
    out[tOk & ~dOk] = scoreTraj[tOk & ~dOk]
    out[dOk & ~tOk] = scoreDeriv[dOk & ~tOk]
    return out


def selectBestIteration_fn(d, traj=0, tolerance=0.02, smoothWindow=5, useValIfAvailable=True):
    """
    Выбрать итерацию по баллу. Правило воспроизводит то, что делает человек по мозаике: найти
    плато лучшего качества и взять в нём самую разреженную модель.

    1. Считаем балл каждой итерации ('computeIterationScores_fn') и отбрасываем непригодные
       (несвежие FoM, заглушки, переменные без единого активного функционала).
    2. Робастифицируем балл: scoreRobust = sqrt(score * медиана по 'smoothWindow' соседям).
       Одиночный выброс (высокий балл в окружении провалов) так теряет половину преимущества,
       а честный максимум на плато сохраняется почти целиком. Просто сглаживать медианой нельзя:
       она сдвигает выбор с резкого, но настоящего пика — а он как раз возникает, когда качество
       на валидации начинает падать сразу после оптимума.
    3. Плато = итерации, у которых робастный балл не ниже максимума минус 'tolerance'
       (аналог правила одной стандартной ошибки).
    4. Внутри плато берём модель с наименьшим числом активных функционалов; при равенстве —
       с наибольшим баллом.

    Parameters
    ----------
    d : dict результатов запуска.
    traj : int. Номер обучающей траектории.
    tolerance : float. Насколько балл может уступать максимуму, чтобы итерация считалась «не хуже».
    smoothWindow : int. Ширина окна скользящей медианы по итерациям.
    useValIfAvailable : bool.

    Returns
    -------
    dict: 'bestRow' (индекс строки в историях), 'bestIter' (номер итерации для
    'plotSelectedIterations'), 'score', 'scoreRobust', 'nActiveTotal', 'plateauRows',
    'verdict' (строка), 'scores' (полный результат 'computeIterationScores_fn').
    """
    s = computeIterationScores_fn(d, traj=traj, useValIfAvailable=useValIfAvailable)

    score = s['score'].copy()
    score[~s['isValid']] = np.nan
    med = _rollingMedian_fn(score, smoothWindow)
    smoothed = np.sqrt(np.clip(score, 0, None) * np.clip(med, 0, None))
    smoothed[~s['isValid']] = np.nan

    if not np.any(np.isfinite(smoothed)):
        return {'bestRow': None, 'bestIter': None, 'score': np.nan, 'scoreRobust': np.nan,
                'nActiveTotal': None, 'plateauRows': np.array([], dtype=int),
                'verdict': 'Ни одна итерация не пригодна для выбора (нет свежих FoM).',
                'scores': s, 'tolerance': tolerance}

    best = np.nanmax(smoothed)
    plateau = np.where(smoothed >= best - tolerance)[0]

    nAct = s['nActiveTotal']
    order = sorted(plateau, key=lambda i: (nAct[i], -smoothed[i]))
    bestRow = int(order[0])

    chosenScore = float(s['score'][bestRow])
    if chosenScore >= 0.5:
        verdict = ('Балл %.3f >= 0.5: модель на этой итерации по совокупности FoM не хуже уровня '
                   '«приличной» по порогам самого тулкита.' % chosenScore)
    else:
        verdict = ('Балл %.3f < 0.5: даже лучшая итерация не дотягивает до порогов «приличной» '
                   'модели из fomChangesDict. Библиотеку/данные нужно менять, а не итерацию.'
                   % chosenScore)

    return {
        'bestRow': bestRow,
        'bestIter': int(s['whichIter'][bestRow]),
        'score': chosenScore,
        'scoreRobust': float(smoothed[bestRow]),
        'nActiveTotal': int(nAct[bestRow]),
        'plateauRows': plateau,
        'verdict': verdict,
        'scores': s,
        'tolerance': tolerance,
    }


def printIterationReport_fn(sel, d=None, topN=10, file=sys.stdout):
    """Текстовый отчёт вместо мозаики: сводка по запуску, топ итераций, разбор выбранной."""
    s = sel['scores']
    score, nAct, valid = s['score'], s['nActiveTotal'], s['isValid']

    print('=' * 96, file=file)
    print('Выбор итерации по скалярному баллу (замена мозаики FoM)', file=file)
    print('=' * 96, file=file)
    print('Итераций записано: %d, пригодны для выбора: %d (несвежие FoM: %d, заглушки: %d)'
          % (len(score), valid.sum(), s['isStale'].sum(), s['isDummy'].sum()), file=file)
    print('Переменных: %d, функционалов в библиотеке: %d' % (s['numVars'], s['numFunctionals']),
          file=file)
    if s['droppedFoms']:
        print('Исключены как константные (не несут информации): %s' % ', '.join(s['droppedFoms']),
              file=file)
    if np.all(np.isnan(s['scoreVal'])):
        print('Валидационные FoM пусты — балл считается только по обучающей траектории.', file=file)
    print('', file=file)

    variableNames = None
    if d is not None and 'variableNames' in d:
        variableNames = list(d['variableNames'])

    # Топ итераций по баллу, но показываем и разреженность, чтобы решение было прозрачным.
    rank = [i for i in np.argsort(-np.where(valid, score, -np.inf)) if valid[i]][:topN]
    compKeys = [k for k in SHORT_NAMES.values() if k in s['components']]
    header = '%6s %8s %9s %8s %8s   ' % ('iter', 'score', 'nFnalsTot', 'traj', 'deriv')
    header += ' '.join('%9s' % k for k in compKeys)
    print('Топ-%d итераций по баллу:' % len(rank), file=file)
    print(header, file=file)
    for i in rank:
        row = '%6d %8.3f %9d %8.3f %8.3f   ' % (s['whichIter'][i], score[i], nAct[i],
                                                s['scoreTraj'][i], s['scoreDeriv'][i])
        row += ' '.join('%9.3f' % s['components'][k][i] for k in compKeys)
        print(row, file=file)
    print('', file=file)

    if sel['bestRow'] is None:
        print(sel['verdict'], file=file)
        return

    i = sel['bestRow']
    print('-' * 96, file=file)
    print('ВЫБРАНА ИТЕРАЦИЯ %d' % sel['bestIter'], file=file)
    print('  балл %.3f (робастный %.3f), плато шириной %d итераций при допуске %.3f'
          % (sel['score'], sel['scoreRobust'], len(sel['plateauRows']), sel['tolerance']),
          file=file)
    print('  активных функционалов: %d всего, до %d на переменную'
          % (nAct[i], s['nActiveMax'][i]), file=file)
    print('  %s' % sel['verdict'], file=file)
    print('  разбор по FoM: %s'
          % ', '.join('%s %.3f' % (k, s['components'][k][i]) for k in compKeys), file=file)

    # Худшие переменные на выбранной итерации — то, что в мозаике искали по цвету точек.
    q = s['perVarQuality'][i]
    worst = np.argsort(q)[:5]
    names = [variableNames[v] if variableNames else 'var%d' % v for v in worst]
    print('  худшие переменные: %s'
          % ', '.join('%s %.2f' % (n, q[v]) for n, v in zip(names, worst)), file=file)
    print('-' * 96, file=file)


def plotIterationScore_fn(sel, figFilename=None):
    """Один график вместо мозаики: балл и число активных функционалов по итерациям, с отметкой
    выбранной итерации. Диагностика, а не инструмент выбора — решение принимает балл."""
    from matplotlib import pyplot as plt

    s = sel['scores']
    x = s['whichIter']
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, np.where(s['isValid'], s['score'], np.nan), '.', color='C0', ms=4, label='балл')
    ax.axhline(0.5, color='k', ls='--', lw=1, label='порог приличия (0.5)')
    if sel['bestRow'] is not None:
        ax.axvline(sel['bestIter'], color='C3', lw=1.5, label='выбор: iter %d' % sel['bestIter'])
    ax.set_xlabel('итерация')
    ax.set_ylabel('балл')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x, s['nActiveTotal'], color='C7', lw=1, alpha=0.6)
    ax2.set_ylabel('активных функционалов, всего', color='C7')

    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    if figFilename:
        fig.savefig(figFilename, dpi=150, bbox_inches='tight')
        print('Saved iteration-score plot to ' + figFilename)
    return fig


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1], 'rb') as f:
        results = pickle.load(f)
    for t in range(int(results['numTrajTrain'])):
        print('\n### Обучающая траектория %d' % t)
        selection = selectBestIteration_fn(results, traj=t)
        printIterationReport_fn(selection, results)
