import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pykalman import KalmanFilter

def svd_smoothing(X, threshold=0.99, r=None, verbose=False):
    # --- 1. SVD разложение ---
    U, S, Vt = np.linalg.svd(X.T, full_matrices=False)

    # --- 2. Выбор числа компонент по порогу объяснённой дисперсии ---
    explained_variance = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(explained_variance)

    if r is None:
        r = np.searchsorted(cumulative_variance, threshold) + 1
        
    print(f'Исходная размерность: {X.shape[0]}')
    print(f'Редуцированная размерность: {r}')
    print(f'Объяснённая дисперсия при r={r}: {cumulative_variance[r-1]:.4f}')

    # --- 3. Усечённые матрицы ---
    U_r = U[:, :r]    # (m, r) — временные коэффициенты
    S_r = S[:r]        # (r,)   — сингулярные значения
    Vt_r = Vt[:r, :]   # (r, n) — главные моды (пространственные)

    # --- 4. Проекция данных в пространство меньшей размерности ---
    # Строки V_r^T — это базисные векторы нового пространства
    # Проецируем X на первые r главных направлений:
    V_r = Vt_r.T       # (n, r) — матрица проекции
    X_reduced = V_r.T @ X  # (r, m) — данные в редуцированном пространстве

    # --- 5. Восстановление (аппроксимация) в исходном пространстве ---
    X_approx = V_r @ X_reduced  # (n, m)

    # --- 6. Визуализация ---
    if verbose:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 6a. Спектр сингулярных значений
        axes[0, 0].bar(range(1, len(S) + 1), S, color='steelblue')
        axes[0, 0].axvline(x=r + 0.5, color='red', linestyle='--', label=f'r = {r}')
        axes[0, 0].set_xlabel('Компонента')
        axes[0, 0].set_ylabel('Сингулярное значение')
        axes[0, 0].set_title('Спектр сингулярных значений')
        axes[0, 0].legend()

        # 6b. Кумулятивная дисперсия
        axes[0, 1].plot(range(1, len(S) + 1), cumulative_variance, 'o-', color='orange')
        axes[0, 1].axhline(y=threshold, color='red', linestyle='--', label=f'{threshold*100:.0f}%')
        axes[0, 1].axvline(x=r, color='green', linestyle='--', label=f'r = {r}')
        axes[0, 1].set_xlabel('Число компонент')
        axes[0, 1].set_ylabel('Доля объяснённой дисперсии')
        axes[0, 1].set_title('Кумулятивная дисперсия')
        axes[0, 1].legend()

        # 6c. Редуцированные координаты во времени
        for i in range(min(r, 5)):
            axes[1, 0].plot(X_reduced[i, :], label=f'Мода {i+1}')
        axes[1, 0].set_xlabel('Время (индекс)')
        axes[1, 0].set_ylabel('Амплитуда')
        axes[1, 0].set_title(f'Динамика в редуцированном пространстве (r={r})')
        axes[1, 0].legend()

        # 6d. Ошибка аппроксимации
        error = np.linalg.norm(X - X_approx, axis=0)
        axes[1, 1].plot(error, color='crimson')
        axes[1, 1].set_xlabel('Время (индекс)')
        axes[1, 1].set_ylabel('||X - X_approx||')
        axes[1, 1].set_title('Ошибка восстановления по времени')

        plt.tight_layout()
        plt.show()

    return X_approx, X_reduced, r


def window_smoothing(trajectories_array, window_size=5):
    trajectories_smooth = uniform_filter1d(
        trajectories_array,
        size=window_size,
        axis=0,                                        # ось времени
        mode='nearest'                                 # граничные значения: ближайший сосед
        # mode='reflect'                               # альтернатива: отражение
        # mode='constant', cval=0.0                    # альтернатива: дополнение нулями
    )
    return trajectories_smooth


def hamming_smoothing(trajectories_array, window_size=5):
    """
    Сглаживание с помощью окна Хэмминга.
    
    Параметры:
    -----------
    trajectories_array : np.ndarray
        Входные данные формы (n_points, n_trajectories) или (n_points,)
    window_size : int
        Размер окна сглаживания (должно быть нечётным)
    
    Возвращает:
    -----------
    np.ndarray
        Сглаженные данные той же формы, что и входные
    """
    # Обеспечиваем нечётный размер окна
    if window_size % 2 == 0:
        window_size += 1
    
    # Создаём окно Хэмминга
    hamming_window = np.hamming(window_size)
    hamming_window /= np.sum(hamming_window)  # Нормализуем
    
    # Обрабатываем одномерный случай
    if trajectories_array.ndim == 1:
        trajectories_array = trajectories_array.reshape(-1, 1)
        was_1d = True
    else:
        was_1d = False
    
    # Применяем свёртку по каждой траектории
    n_points, n_trajectories = trajectories_array.shape
    trajectories_smooth = np.zeros_like(trajectories_array)
    
    half_window = window_size // 2
    
    for i in range(n_trajectories):
        # Свёртка с отражением границ
        padded = np.pad(trajectories_array[:, i], (half_window, half_window), mode='reflect')
        for j in range(n_points):
            trajectories_smooth[j, i] = np.sum(padded[j:j + window_size] * hamming_window)
    
    if was_1d:
        trajectories_smooth = trajectories_smooth.flatten()
    
    return trajectories_smooth


def kalman_smoothing(trajectories_array, 
                     process_variance=1e-5, 
                     measurement_variance=1e-2,
                     initial_state=None,
                     initial_covariance=1.0):
    """
    Сглаживание с помощью фильтра Калмана.
    
    Параметры:
    -----------
    trajectories_array : np.ndarray
        Входные данные формы (n_points, n_trajectories) или (n_points,)
    process_variance : float
        Дисперсия процесса (Q) - насколько быстро меняется состояние
    measurement_variance : float
        Дисперсия измерений (R) - уровень шума в данных
    initial_state : float, optional
        Начальное состояние (по умолчанию первое значение)
    initial_covariance : float
        Начальная ковариация
    
    Возвращает:
    -----------
    np.ndarray
        Сглаженные данные той же формы, что и входные
    """
    # Обрабатываем одномерный случай
    if trajectories_array.ndim == 1:
        trajectories_array = trajectories_array.reshape(-1, 1)
        was_1d = True
    else:
        was_1d = False
    
    n_points, n_trajectories = trajectories_array.shape
    trajectories_smooth = np.zeros_like(trajectories_array)
    
    # Инициализация фильтра Калмана для каждой траектории
    for i in range(n_trajectories):
        measurements = trajectories_array[:, i]
        
        # Инициализация состояния
        if initial_state is None:
            x = measurements[0]
        else:
            x = initial_state
        
        P = initial_covariance  # Ковариация
        Q = process_variance    # Дисперсия процесса
        R = measurement_variance  # Дисперсия измерений
        
        smoothed = np.zeros(n_points)
        
        # Прямой проход (фильтрация)
        for t in range(n_points):
            # Предсказание
            x_pred = x
            P_pred = P + Q
            
            # Обновление
            K = P_pred / (P_pred + R)  # Коэффициент Калмана
            x = x_pred + K * (measurements[t] - x_pred)
            P = (1 - K) * P_pred
            
            smoothed[t] = x
        
        # Обратный проход (сглаживание RTS - Rauch-Tung-Striebel)
        for t in range(n_points - 2, -1, -1):
            # Предсказание ковариации вперёд
            P_pred_next = P + Q
            
            # Коэффициент сглаживания
            A = P / P_pred_next
            
            # Коррекция сглаженного значения
            smoothed[t] = smoothed[t] + A * (smoothed[t + 1] - smoothed[t])
        
        trajectories_smooth[:, i] = smoothed
    
    if was_1d:
        trajectories_smooth = trajectories_smooth.flatten()
    
    return trajectories_smooth
