import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from scipy.integrate import solve_ivp
from core.plot import plot_trajectory_comparison_nd
from toolkitSupportFunctions import (boundednessViolation_fn, stabilizeLinearModel_fn,
                                     trappingLinearModel_fn)

def fit_polynomial_ode(
    trajectory: np.ndarray,
    time: np.ndarray,
    poly_degree: int = 2,
    visualize: bool = True,
    report: bool = True,
    extend_time: np.ndarray = None,
    integration_subdivisions: int = 1,
    coord_names: list = None,
    bounded: bool = False,
    boundedness_margin: float = 0.0,
    strict_boundedness_margin: float = -1e-3,
    boundedness_norm_type: str = 'spectral',
    boundedness_method: str = 'trapping',
    boundedness_repair_metric: str = 'symmetric',
    boundedness_ridge_ladder: tuple = (1., 10., 100., 1e3, 1e4),
    boundedness_nu_ladder: tuple = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
    boundedness_max_num_free_entries: float = float('inf'),
) -> dict:
    """
    Проверяет, насколько хорошо переданная траектория описывается ОДУ
    с полиномиальной правой частью заданной степени.

    Метод:
    - Численно вычисывает производные траектории (dx/dt)
    - Строит полиномиальные признаки из координат x
    - Обучает линейную регрессию: dx/dt ≈ P(x)
    - Интегрирует найденное ОДУ и сравнивает с исходной траекторией

    Parameters
    ----------
    trajectory : np.ndarray, shape (n_points, n_dims)
        Наблюдаемая траектория.
    time : np.ndarray, shape (n_points,)
        Временные метки (не обязательно равномерные).
    poly_degree : int
        Степень полиномов в правой части ОДУ.
    visualize : bool
        Отображать ли графики сравнения.
    report : bool
        Выводить ли подробный отчет.
    extend_time : np.ndarray, shape (n_extended,), optional
        Временные метки для продления предсказанной траектории за пределы
        обучающего интервала. Если указано, интегрирование будет выполнено
        на расширенном временном интервале.
    integration_subdivisions : int
        Число отрезков, на которые разбивается каждый интервал между
        соседними точками t_eval при интегрировании. Значение по умолчанию 1
        означает интегрирование только в точках t_eval. Значение > 1 позволяет
        получить более детальную сетку для интегрирования, что может повысить
        точность численного решения ОДУ.
    coord_names : list of str, optional
        Список названий координат для визуализации и отчета. Если None,
        используются ['x0', 'x1', 'x2', ...].
    bounded : bool
        Фитить ли заведомо ОГРАНИЧЕННУЮ систему. Реализовано только для
        poly_degree=1, при другой степени поднимается ValueError. Модель тогда
        имеет вид dx/dt = c + A*x, и на линейный блок A накладывается условие
        трэппинга A'P + PA <= 2*gamma*P (gamma <= 0) для некоторой P > 0: тогда
        V(x) = x'P*x не растет и траектория заперта в множестве уровня V, то есть
        проинтегрированная модель гарантированно не улетает. Каким именно способом
        ограничение накладывается, выбирает boundedness_method.
        ВНИМАНИЕ: функция ничего не сглаживает — производные считаются по тому, что
        передали. Нужно ли ограничение, зависит именно от этого. На сырых данных
        лояльности неограниченная модель улетает и ограничение обязательно; после
        сглаживания (в ноутбуках hamming_smoothing(window_size=19) и подобные) спектр
        поджимается примерно в 10 раз, и на горизонте самих данных ограничение уже скорее
        отнимает точность, чем добавляет. При этом max Re eig(A) остаётся положительным,
        поэтому за горизонтом ~1500-3000 шагов модель всё равно расходится, и там
        ограничение — единственное, что держит её конечной. Таблицы по обоим режимам и
        по горизонтам — в CLAUDE.md, раздел 'Known issues'.
    boundedness_margin : float, <= 0
        gamma. 0 -> ограниченность на грани, чисто мнимые собственные значения
        (незатухающие осцилляторы) остаются допустимыми. ВНИМАНИЕ: константа в
        библиотеке активна всегда (include_bias=True), а аффинный член может расти
        вдоль ядра sym(A), поэтому при gamma=0 фактически используется
        strict_boundedness_margin. Радиус трэппинг-шара тогда ||c|| / |gamma|.
    strict_boundedness_margin : float, < 0
        gamma, подставляемая вместо boundedness_margin, когда та равна 0, а
        константа активна (то есть всегда, см. выше).
    boundedness_norm_type : {'spectral', 'identity'}
        Как обходиться с весом P в условии трэппинга.
        'spectral' -> P свободна, условие сводится к max Re eig(A) <= gamma. Для
        чисто линейной модели это точное условие ограниченности, поэтому оно и по
        умолчанию.
        'identity' -> P = I, буквальное условие trapping-SINDy max eig sym(A) <= gamma.
        Достаточное, но не необходимое, и сильно консервативное для ненормальных A.
    boundedness_method : {'trapping', 'ladder'}
        Чем ремонтировать модель, когда ограничение нарушено. Оба варианта
        возвращают модель, которая ему удовлетворяет.
        'trapping' (по умолчанию) -> 'trappingLinearModel_fn', relax-and-split из
        Kaptanoglu et al. 2021 в линейной специализации: чередуем совместное решение
        по всем активным элементам с перепроекцией вспомогательной матрицы из
        ТЕКУЩЕЙ sym(A), отжигая boundedness_nu_ladder вниз до выполнимости. Держит
        фит производных заметно ближе к неограниченному (медианный R² +0.15 против
        -12.8 ladder'а на синтетических линейных системах), но стоит примерно в 46
        раз дороже и траектории там не улучшил — см. 'benchmark_trapping_vs_ladder.py'.
        'ladder' -> 'stabilizeLinearModel_fn', четыре шага: уже ограничена ->
        ridge-регрессия с притяжением к спроецированной цели (boundedness_ridge_ladder)
        -> проекция на {sym(A) <= gamma} -> диагональный сдвиг. Дешево.
        ВНИМАНИЕ: 'trapping' решает плотную систему размера "число активных элементов
        библиотеки", а здесь support плотный, то есть n_dims*(1+n_dims). По умолчанию
        boundedness_max_num_free_entries = inf, то есть откат на ladder не срабатывает
        сам по себе — для 87 признаков лояльности это плотная система 7656x7656, и
        она решается как есть, сколько бы это ни стоило. Откат на ladder ('boundedness_action'
        с префиксом 'ladderFallback:') остаётся доступен, если явно понизить
        boundedness_max_num_free_entries.
    boundedness_repair_metric : {'symmetric', 'schur', 'lyapunov'}
        В какой метрике метод 'ladder' ЧИНИТ модель — ось, независимая от
        boundedness_norm_type, который говорит, что ПРОВЕРЯЕТСЯ. Полное описание в
        докстринге 'stabilizeLinearModel_fn'.
        'symmetric' (по умолчанию) -> исходное поведение: цель ridge'а — проекция
        симметричной части, шаг проекции — на {sym(A) <= gamma}.
        'schur' -> цель ridge'а строится клемпингом спектра через разложение Шура, то
        есть двигаются только нарушающие собственные значения, а собственные векторы и
        вся ненормальная структура A остаются. Симметричная цель сохраняется запасной,
        поэтому результат не бывает хуже 'symmetric'.
        'lyapunov' -> то же плюс взвешенная проекция на {A'P + PA <= 2*gamma*P}.
        Работают только при boundedness_norm_type='spectral'; при 'identity'
        понижаются до 'symmetric' с сообщением об ошибке.
        ИЗМЕРЕНО: на траекториях обе проигрывают. На лояльности (70% обучения,
        hamming 19, свободный прогон по отложенному хвосту — '_probe_repair_metric_loyalty.py')
        'schur' даёт v3/87 rmseTail 0.039 -> 72.5 при пике |x| 579 от размаха
        данных, continuous/26 rmseTail 0.230 -> 2.44 — и это при том, что R²
        производных на continuous улучшается в 3000 раз. Причина в том, что P = I
        гарантирует ||x(t)|| <= ||x0||, а спектральное условие допускает
        транзиентный рост, который клемпинг Шура как раз бережно сохраняет.
        'lyapunov' ни разу не отличился от 'schur'. Поэтому по умолчанию
        'symmetric'; 'schur' имеет смысл, только если целевая метрика — фит
        производных, а не прогноз.
    boundedness_ridge_ladder : tuple of float, возрастающая
        Лестница коэффициентов L2-притяжения к стабилизированной цели для метода
        'ladder'. Побеждает первая ступень, на которой ограничение выполняется.
        Используется и как fallback метода 'trapping'.
    boundedness_nu_ladder : tuple of float, УБЫВАЮЩАЯ
        Силы релаксации для метода 'trapping', проходятся с warm start. Меньшая nu
        тянет сильнее. Для normType='identity' выполнимость наступает только около
        1e-6, для 'spectral' — на несколько ступеней раньше.
    boundedness_max_num_free_entries : float
        Порог, выше которого 'trapping' отказывается от задачи и управление уходит
        на ladder. По умолчанию inf — то есть порога нет и trapping решает задачу
        любого размера.

    Returns
    -------
    results : dict
        Словарь с ключами:
        - 'r2_derivative': R² качество фита производных (до интегрирования), shape (n_dims,)
        - 'r2_trajectory': R² качество восстановленной траектории по координатам, shape (n_dims,)
        - 'rmse_trajectory': RMSE по восстановленной траектории по координатам, shape (n_dims,)
        - 'rmse_total'   : Общая RMSE для всей многомерной траектории (скаляр)
        - 'r2_trajectory_box': Метрики для box_plot по R² (min, max, median, mean, std, q1, q3)
        - 'rmse_trajectory_box': Метрики для box_plot по RMSE (min, max, median, mean, std, q1, q3)
        - 'predicted_traj': восстановленная траектория shape (n_points, n_dims) или shape (n_extended, n_dims) при extend_time
        - 'predicted_traj_train': восстановленная траектория на обучающем интервале shape (n_points, n_dims), только при extend_time
        - 'time_used': временные метки использованные для интегрирования. При integration_subdivisions > 1 содержит уплотнённую сетку shape (n_points + (n_points-1)*(integration_subdivisions-1),) или (n_extended + (n_extended-1)*(integration_subdivisions-1),) при extend_time
        - 'coefficients' : список массивов коэффициентов регрессии
        - 'feature_names': названия полиномиальных признаков

        Только при poly_degree=1 (модель dx/dt = c + A*x):
        - 'boundedness_violation_unconstrained': насколько неограничена модель ДО
          применения ограничения, в норме boundedness_norm_type. <= gamma означает,
          что ограничение и так выполнено
        - 'boundedness_violation': то же ПОСЛЕ. Совпадает с предыдущим при bounded=False
        - 'boundedness_action': что именно сработало. Для 'ladder' — 'already-bounded',
          'ridge(lam=...)', 'projection', 'diagonalShift'; для 'trapping' —
          'already-bounded', 'trapping(nu=...)', 'projection', 'diagonalShift', либо
          'ladderFallback:<шаг ladder>', когда задача оказалась слишком большой.
          None при bounded=False
        - 'boundedness_gamma': фактически использованная gamma
        - 'boundedness_method': запрошенный метод ремонта, либо None при bounded=False
    """
    trajectory = np.asarray(trajectory, dtype=float)
    time = np.asarray(time, dtype=float)

    n_points, n_dims = trajectory.shape

    # ------------------------------------------------------------------
    # 0. Проверка параметров ограниченности и названия координат
    # ------------------------------------------------------------------
    if bounded and poly_degree != 1:
        raise ValueError(
            "bounded=True реализовано только для poly_degree=1: условие трэппинга "
            "накладывается на линейный блок A модели dx/dt = c + A*x, а при степени "
            f"{poly_degree} в библиотеке есть члены степени >= 2. Получено poly_degree={poly_degree}."
        )
    if bounded and boundedness_margin > 0:
        raise ValueError(
            "boundedness_margin (gamma) должна быть <= 0, получено "
            f"{boundedness_margin}."
        )
    if bounded and boundedness_norm_type not in ('spectral', 'identity'):
        raise ValueError(
            "boundedness_norm_type должен быть 'spectral' или 'identity', получено "
            f"{boundedness_norm_type!r}."
        )
    if bounded and boundedness_method not in ('trapping', 'ladder'):
        raise ValueError(
            "boundedness_method должен быть 'trapping' или 'ladder', получено "
            f"{boundedness_method!r}."
        )

    # Имена нужны до фита: ремонт ограниченности пишет их в лог, когда вынужден
    # восстановить культнутый само-член.
    if coord_names is None:
        coord_names = [f"x{i}" for i in range(n_dims)]

    # ------------------------------------------------------------------
    # 1. Численное дифференцирование (центральные разности)
    # ------------------------------------------------------------------
    derivatives = _numerical_derivatives(trajectory, time)   # (n_points, n_dims)

    # ------------------------------------------------------------------
    # 2. Полиномиальные признаки
    # ------------------------------------------------------------------
    poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
    X_poly = poly.fit_transform(trajectory)                  # (n_points, n_features)
    feature_names = poly.get_feature_names_out(
        [f"x{i}" for i in range(n_dims)]
    )

    # ------------------------------------------------------------------
    # 3. Линейная регрессия для каждой координаты
    # ------------------------------------------------------------------
    coefficients = []

    for dim in range(n_dims):
        y = derivatives[:, dim]
        model = LinearRegression(fit_intercept=False)  # bias уже в poly (include_bias=True)
        # model = Lasso(fit_intercept=True, alpha=0.0)
        model.fit(X_poly, y)

        coefficients.append(model.coef_)

    coeff_matrix = np.asarray(coefficients)                  # (n_dims, n_features)

    # ------------------------------------------------------------------
    # 3b. Ограниченность (только для линейной модели dx/dt = c + A*x)
    # ------------------------------------------------------------------
    # Раскладка колонок при poly_degree=1 -- ['1', 'x0', ..., 'x{n-1}'] -- ровно та, которую ждут
    # функции ремонта: c = coeffArray[:, 0], A = coeffArray[:, 1:1+numVars].
    boundedness_violation_unconstrained = None
    boundedness_violation = None
    boundedness_action = None
    boundedness_gamma = None

    if poly_degree == 1:
        boundedness_violation_unconstrained = boundednessViolation_fn(
            coeff_matrix[:, 1:1 + n_dims], boundedness_norm_type
        )
        boundedness_violation = boundedness_violation_unconstrained

    if bounded:
        # Константа активна всегда, поэтому gamma ужесточается внутри функций ремонта;
        # повторяем ту же логику, чтобы вернуть фактическое значение.
        boundedness_gamma = strict_boundedness_margin if boundedness_margin == 0 \
            else boundedness_margin

        # Здесь нет куллинга, поэтому support плотный, а веса точек единичные -- это дает
        # ровно невзвешенный МНК на шаге ridge-регрессии.
        functions_to_use = np.ones(coeff_matrix.shape, dtype=bool)
        design_matrices = {
            v: {'X': X_poly,
                'y': derivatives[:, v],
                'sampleWeights': np.ones(n_points),
                'functionInds': np.arange(X_poly.shape[1])}
            for v in range(n_dims)
        }
        # outputFilename нужен функциям только для лога восстановленных само-членов и
        # диагонального сдвига; при плотном support восстанавливать нечего, а про сдвиг
        # рассказывает возвращаемое 'boundedness_action'.
        def stabilize_with_ladder():
            return stabilizeLinearModel_fn(
                coeff_matrix, functions_to_use, design_matrices, n_dims,
                boundedness_margin, strict_boundedness_margin, boundedness_ridge_ladder,
                coord_names, os.devnull, boundedness_norm_type, boundedness_repair_metric,
            )

        if boundedness_method == 'trapping':
            new_coeff_matrix, _, boundedness_action, _ = trappingLinearModel_fn(
                coeff_matrix, functions_to_use, design_matrices, n_dims,
                boundedness_margin, strict_boundedness_margin, boundedness_nu_ladder,
                coord_names, os.devnull, boundedness_norm_type,
                maxNumFreeEntries=boundedness_max_num_free_entries,
            )
            # Support плотный, то есть numFree = n_dims*(1+n_dims), и на широких библиотеках
            # (лояльность: 87 переменных -> 7656) trapping откажется от задачи. Возвращать при
            # этом неограниченную модель нельзя -- bounded=True обещает обратное, -- поэтому
            # управление уходит на ladder, а факт отката виден в 'boundedness_action'.
            if boundedness_action == 'skipped-too-large':
                new_coeff_matrix, _, ladder_action, _ = stabilize_with_ladder()
                boundedness_action = 'ladderFallback:' + ladder_action
            coeff_matrix = new_coeff_matrix
        else:
            coeff_matrix, _, boundedness_action, _ = stabilize_with_ladder()
        coefficients = [row for row in coeff_matrix]
        boundedness_violation = boundednessViolation_fn(
            coeff_matrix[:, 1:1 + n_dims], boundedness_norm_type
        )

    # R^2 фита производных считается по итоговым коэффициентам, чтобы была видна цена
    # ограничения:
    r2_derivative = np.array([
        r2_score(derivatives[:, dim], X_poly @ coeff_matrix[dim])
        for dim in range(n_dims)
    ])

    # ------------------------------------------------------------------
    # 4. Интегрирование найденного ОДУ
    # ------------------------------------------------------------------
    # Правая часть строится по матрице коэффициентов, а не по объектам sklearn: иначе
    # ограниченные коэффициенты не попали бы в интегрирование.
    def ode_rhs(t: float, x: np.ndarray) -> np.ndarray:
        return coeff_matrix @ poly.transform(x.reshape(1, -1))[0]

    x0 = trajectory[0]
    
    # Определяем временной интервал для интегрирования
    if extend_time is not None:
        extend_time = np.asarray(extend_time, dtype=float)
        # Интегрируем на расширенном интервале
        t_span = (time[0], max(time[-1], extend_time[-1]))
        t_eval_base = extend_time
    else:
        t_span = (time[0], time[-1])
        t_eval_base = time

    # Создаём более мелкую сетку для интегрирования, если integration_subdivisions > 1
    if integration_subdivisions > 1:
        t_eval_fine = _create_fine_time_grid(t_eval_base, integration_subdivisions)
    else:
        t_eval_fine = t_eval_base

    sol = solve_ivp(
        ode_rhs,
        t_span,
        x0,
        t_eval=t_eval_fine,
        method="RK45",
        rtol=1e-8,
        atol=1e-9,
    )

    if sol.success:
        predicted_traj_fine = sol.y.T                        # (n_fine_points, n_dims)
    else:
        print(f"⚠️  solve_ivp завершился с ошибкой: {sol.message}")
        predicted_traj_fine = np.full((len(t_eval_fine), n_dims), np.nan)

    # Если использовалась мелкая сетка, интерполируем обратно на исходную
    if integration_subdivisions > 1:
        # Предсказание на исходной сетке (для возврата)
        predicted_traj = _interpolate_to_grid(
            predicted_traj_fine, integration_subdivisions
        )
    else:
        predicted_traj = predicted_traj_fine

    # Если используется extend_time, сохраняем также предсказание на обучающем интервале
    if extend_time is not None:
        # Получаем предсказание на обучающем интервале для метрик
        # Если integration_subdivisions > 1, создаём мелкую сетку и для обучающего интервала
        if integration_subdivisions > 1:
            t_train_fine = _create_fine_time_grid(time, integration_subdivisions)
        else:
            t_train_fine = time
            
        sol_train = solve_ivp(
            ode_rhs,
            (time[0], time[-1]),
            x0,
            t_eval=t_train_fine,
            method="RK45",
            rtol=1e-8,
            atol=1e-9,
        )
        if sol_train.success:
            predicted_traj_train_fine = sol_train.y.T
            # Интерполируем на исходную сетку времени для метрик
            predicted_traj_train = _interpolate_to_grid(
                predicted_traj_train_fine, integration_subdivisions
            )
        else:
            print(f"⚠️  solve_ivp на обучающем интервале завершился с ошибкой: {sol_train.message}")
            predicted_traj_train = np.full_like(trajectory, np.nan)
    else:
        # Без extend_time, predicted_traj_train = predicted_traj для метрик
        predicted_traj_train = predicted_traj

    # ------------------------------------------------------------------
    # 5. Метрики качества траектории
    # ------------------------------------------------------------------
    # Метрики вычисляются на обучающем интервале
    r2_traj = np.array([
        r2_score(trajectory[:, d], predicted_traj_train[:, d])
        for d in range(n_dims)
    ])
    rmse_traj = np.sqrt(np.mean((trajectory - predicted_traj_train) ** 2, axis=0))
    
    # Общая RMSE для всей многомерной траектории
    rmse_total = np.sqrt(np.mean((trajectory - predicted_traj_train) ** 2))
    
    # Метрики для box_plot по R² и RMSE
    r2_traj_box = {
        'min': float(np.min(r2_traj)),
        'max': float(np.max(r2_traj)),
        'median': float(np.median(r2_traj)),
        'q1': float(np.percentile(r2_traj, 25)),
        'q3': float(np.percentile(r2_traj, 75)),
        'mean': float(np.mean(r2_traj)),
        'std': float(np.std(r2_traj)),
    }
    rmse_traj_box = {
        'min': float(np.min(rmse_traj)),
        'max': float(np.max(rmse_traj)),
        'median': float(np.median(rmse_traj)),
        'q1': float(np.percentile(rmse_traj, 25)),
        'q3': float(np.percentile(rmse_traj, 75)),
        'mean': float(np.mean(rmse_traj)),
        'std': float(np.std(rmse_traj)),
    }

    # ------------------------------------------------------------------
    # 6. Вывод результатов
    # ------------------------------------------------------------------
    if report:
        _print_report(
            n_dims, poly_degree, feature_names,
            coefficients, r2_derivative, r2_traj, rmse_traj,
            rmse_total, r2_traj_box, rmse_traj_box, coord_names,
            bounded, boundedness_norm_type, boundedness_gamma,
            boundedness_violation_unconstrained, boundedness_violation,
            boundedness_action, boundedness_method,
        )

    # ------------------------------------------------------------------
    # 7. Визуализация
    # ------------------------------------------------------------------
    if visualize:
        plot_trajectory_comparison_nd(trajectory, predicted_traj_train, time, coord_names=coord_names)

    # ------------------------------------------------------------------
    # 8. Формирование результата
    # ------------------------------------------------------------------
    result = {
        "r2_derivative": r2_derivative,
        "r2_trajectory": r2_traj,
        "rmse_trajectory": rmse_traj,
        "rmse_total": rmse_total,
        "r2_trajectory_box": r2_traj_box,
        "rmse_trajectory_box": rmse_traj_box,
        "predicted_traj": predicted_traj,
        "time_used": t_eval_fine,
        "coefficients": coefficients,
        "feature_names": feature_names,
    }
    
    # Добавляем предсказание на обучающем интервале при extend_time
    if extend_time is not None:
        result["predicted_traj_train"] = predicted_traj_train

    # Диагностика ограниченности имеет смысл только для линейной модели
    if poly_degree == 1:
        result["boundedness_violation_unconstrained"] = boundedness_violation_unconstrained
        result["boundedness_violation"] = boundedness_violation
        result["boundedness_action"] = boundedness_action
        result["boundedness_gamma"] = boundedness_gamma
        result["boundedness_method"] = boundedness_method if bounded else None

    return result


# ──────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────

def _numerical_derivatives(trajectory: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Вычисление производных методом конечных разностей повышенной точности O(h^4)
    
    Использует 5-точечный шаблон для внутренних точек:
    f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    
    Для граничных точек используются односторонние разности 4-го порядка:
    f'(x₀) ≈ (-25f₀ + 48f₁ - 36f₂ + 16f₃ - 3f₄) / (12h)
    f'(xₙ) ≈ (25fₙ - 48fₙ₋₁ + 36fₙ₋₂ - 16fₙ₋₃ + 3fₙ₋₄) / (12h)
    
    Parameters
    ----------
    trajectory : np.ndarray, shape (n_points, n_dims)
        Наблюдаемая траектория.
    time : np.ndarray, shape (n_points,)
        Временные метки (предполагается равномерная сетка).
    
    Returns
    -------
    deriv : np.ndarray, shape (n_points, n_dims)
        Производные траектории.
    """
    n_points, n_dims = trajectory.shape
    
    # Вычисляем средний шаг по времени (предполагаем равномерную сетку)
    dt = np.mean(np.diff(time))
    
    # Инициализируем массив для производных
    deriv = np.zeros_like(trajectory)
    
    # Для очень коротких рядов используем метод 2-го порядка
    if n_points < 5:
        if n_points >= 3:
            deriv[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2 * dt)
            deriv[0] = (trajectory[1] - trajectory[0]) / dt
            deriv[-1] = (trajectory[-1] - trajectory[-2]) / dt
        else:
            deriv[:-1] = (trajectory[1:] - trajectory[:-1]) / dt
            deriv[-1] = deriv[-2]
        return deriv
    
    # 5-точечный шаблон для внутренних точек (O(h^4))
    # f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    deriv[2:-2] = (
        -trajectory[4:] + 
        8 * trajectory[3:-1] - 
        8 * trajectory[1:-3] + 
        trajectory[:-4]
    ) / (12 * dt)
    
    # Граничные точки - односторонние разности 4-го порядка
    # Левая граница
    deriv[0] = (-25 * trajectory[0] + 48 * trajectory[1] - 36 * trajectory[2] + 
                16 * trajectory[3] - 3 * trajectory[4]) / (12 * dt)
    deriv[1] = (-3 * trajectory[0] - 10 * trajectory[1] + 18 * trajectory[2] - 
                6 * trajectory[3] + trajectory[4]) / (12 * dt)
    
    # Правая граница
    deriv[-1] = (25 * trajectory[-1] - 48 * trajectory[-2] + 36 * trajectory[-3] - 
                 16 * trajectory[-4] + 3 * trajectory[-5]) / (12 * dt)
    deriv[-2] = (-trajectory[-5] + 6 * trajectory[-4] - 18 * trajectory[-3] + 
                 10 * trajectory[-2] + 3 * trajectory[-1]) / (12 * dt)
    
    return deriv


def _print_report(
    n_dims, poly_degree, feature_names,
    coefficients, r2_derivative, r2_traj, rmse_traj,
    rmse_total, r2_traj_box, rmse_traj_box, coord_names,
    bounded=False, boundedness_norm_type='spectral', boundedness_gamma=None,
    boundedness_violation_unconstrained=None, boundedness_violation=None,
    boundedness_action=None, boundedness_method='trapping',
):
    """Красивый вывод результатов в консоль."""
    sep = "─" * 60

    print(f"\n{'═'*60}")
    print(f"  Polynomial ODE fit  |  degree={poly_degree}  |  dims={n_dims}"
          f"{f'  |  bounded ({boundedness_method})' if bounded else ''}")
    print(f"{'═'*60}")

    # Ограниченность: имеет смысл только для линейной модели
    if poly_degree == 1:
        norm_label = ("max Re eig(A)" if boundedness_norm_type == 'spectral'
                      else "max eig sym(A)")
        print(f"\n🔒 Ограниченность ({boundedness_norm_type}, {norm_label}):")
        if bounded:
            ok = "✅" if boundedness_violation <= boundedness_gamma else "❌"
            print(f"  gamma        : {boundedness_gamma:+.4e}")
            print(f"  до фикса     : {boundedness_violation_unconstrained:+.4e}")
            print(f"  после фикса  : {boundedness_violation:+.4e}  {ok}")
            print(f"  что сделано  : {boundedness_action}")
        else:
            ok = "✅" if boundedness_violation_unconstrained <= 0 else "❌"
            print(f"  {boundedness_violation_unconstrained:+.4e}  {ok}"
                  f"{'' if boundedness_violation_unconstrained <= 0 else '  (модель разойдется, см. bounded=True)'}")

    # Уравнения (значимые члены)
    print("\n📐 Найденные уравнения (|coef| > 1e-6):")
    for dim in range(n_dims):
        terms = [
            f"{c:+.4f}·{name}"
            for c, name in zip(coefficients[dim], feature_names)
            if abs(c) > 1e-6
        ]
        rhs = "  ".join(terms) if terms else "0"
        print(f"  d{coord_names[dim]}/dt = {rhs}")

    # Качество фита производных
    print(f"\n📊 R² фита производных (до интегрирования):")
    for dim in range(n_dims):
        bar = _r2_bar(r2_derivative[dim])
        print(f"  d{coord_names[dim]}/dt : R²={r2_derivative[dim]:7.4f}  {bar}")

    # Качество восстановленной траектории
    print(f"\n🚀 Качество восстановленной траектории:")
    for dim in range(n_dims):
        bar = _r2_bar(r2_traj[dim])
        print(
            f"  {coord_names[dim]} : R²={r2_traj[dim]:7.4f}  "
            f"RMSE={rmse_traj[dim]:.4e}  {bar}"
        )
    
    # Общая RMSE
    print(f"\n📈 Общая RMSE многомерной траектории: {rmse_total:.4e}")
    
    # Метрики для box_plot
    print(f"\n📊 Метрики для box_plot:")
    print(f"  R² trajectory:")
    print(f"    min={r2_traj_box['min']:.4f}, max={r2_traj_box['max']:.4f}, "
          f"median={r2_traj_box['median']:.4f}, mean={r2_traj_box['mean']:.4f}, "
          f"std={r2_traj_box['std']:.4f}")
    print(f"    Q1={r2_traj_box['q1']:.4f}, Q3={r2_traj_box['q3']:.4f}")
    print(f"  RMSE trajectory:")
    print(f"    min={rmse_traj_box['min']:.4e}, max={rmse_traj_box['max']:.4e}, "
          f"median={rmse_traj_box['median']:.4e}, mean={rmse_traj_box['mean']:.4e}, "
          f"std={rmse_traj_box['std']:.4e}")
    print(f"    Q1={rmse_traj_box['q1']:.4e}, Q3={rmse_traj_box['q3']:.4e}")

    print(f"\n{'═'*60}\n")


def _r2_bar(r2: float, width: int = 20) -> str:
    """Маленький ASCII-прогресс-бар для R²."""
    r2_clipped = max(0.0, min(1.0, r2))
    filled = int(r2_clipped * width)
    color = "✅" if r2_clipped > 0.95 else ("⚠️ " if r2_clipped > 0.7 else "❌")
    return f"[{'█'*filled}{'░'*(width-filled)}] {color}"


def _create_fine_time_grid(t_eval: np.ndarray, n_subdivisions: int) -> np.ndarray:
    """
    Создаёт более мелкую временную сетку для интегрирования ОДУ.
    
    Каждый интервал между соседними точками t_eval разбивается на n_subdivisions
    равных отрезков.
    
    Parameters
    ----------
    t_eval : np.ndarray, shape (n_points,)
        Исходные временные точки.
    n_subdivisions : int
        Число отрезков, на которые разбивается каждый интервал.
        
    Returns
    -------
    t_fine : np.ndarray, shape (n_fine_points,)
        Уплотнённая временная сетка.
    """
    if n_subdivisions < 1:
        raise ValueError("n_subdivisions должен быть >= 1")
    
    if len(t_eval) < 2:
        return t_eval.copy()
    
    n_intervals = len(t_eval) - 1
    n_fine_points = n_intervals * n_subdivisions + 1
    
    t_fine = np.linspace(t_eval[0], t_eval[-1], n_fine_points)
    return t_fine


def _interpolate_to_grid(values: np.ndarray, n_subdivisions: int) -> np.ndarray:
    """
    Извлекает значения с мелкой сетки на исходную сетку.
    
    Точки исходной сетки соответствуют индексам, кратным n_subdivisions.
    
    Parameters
    ----------
    values : np.ndarray, shape (n_from_points, n_dims)
        Значения на уплотнённой сетке.
    n_subdivisions : int
        Число отрезков разбиения (шаг между индексами).
        
    Returns
    -------
    values_interp : np.ndarray, shape (n_to_points, n_dims)
        Значения на целевой сетке.
    """
    # Выбираем каждую n_subdivisions-ную точку
    indices = np.arange(0, len(values), n_subdivisions)
    return values[indices]
