import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = ['#3e82fc', '#fc824a', '#f7d560', '#f773b4', '#01c08d', '#8f99fb']
sns.set_palette(colors)

def plot_trajectory_comparison(true_traj, pred_traj, time):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Временные ряды для каждой координаты
    for i, (ax, coord, ylim) in enumerate(zip(axes.flat[:3], ['X', 'Y', 'Z'], [(-12.5, 17.5), (-15, 22.5), (0, 37.5)])):
        ax.plot(time, true_traj[:,i], label='true')
        ax.plot(time, pred_traj[:,i], label='predicted', linestyle='--')
        ax.set_ylim(ylim)
        ax.set_xlabel('time')
        ax.set_ylabel(coord)
        ax.set_title(f'{coord}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    

    # Ошибка по времени
    ax = axes.flat[3]
    error = np.linalg.norm(true_traj - pred_traj, axis=1)
    ax.plot(time, error, linewidth=2)
    ax.set_xlabel('time')
    ax.set_ylabel('RMSE')
    ax.set_title('Prediction loss')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Training History')
    plt.tight_layout()
    plt.show()


def plot_trajectories(trajectories, baseline, time, dt=0.005, coord_names=None, labels=None):
    """
    Построение графиков траекторий
    
    Parameters:
    -----------
    trajectories : list of np.array
        Список траекторий, каждая shape (n_points, n_dims)
    baseline : np.array
        Базовая траектория shape (n_points, n_dims)
    time : np.array
        Временные метки shape (n_points,), если передать None, время инициируется как np.arange(n_points)*dt
    dt : float
        Расстояние между точками траектории, используется только если time is None
    coord_names : list of str, optional
        Список названий координат, если None, используются ['X', 'Y', 'Z', 'W', 'V', 'U', 'X_6'...]
    labels : list of str, optional
        Список названий траекторий
    """
    n_dims = trajectories[0].shape[1]
    n_points = trajectories[0].shape[0]
    
    if baseline is not None:
        n_plots = n_dims + 1
    else: 
        n_plots = n_dims
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axes = np.array(axes).flatten()
    
    # Названия координат
    if coord_names is None:
        coord_names = ['X', 'Y', 'Z', 'W', 'V', 'U']
        if n_dims > len(coord_names):
            coord_names.extend([f'Dim_{i}' for i in range(len(coord_names), n_dims)])

    if time is None:
        time = np.arange(n_points)*dt
    
    # Строим графики для каждой траектории
    for traj_idx, traj in enumerate(trajectories):
        label = labels[traj_idx] if labels is not None else None
        # Временные ряды для каждой координаты
        for i in range(n_dims):
            ax = axes[i]
            ax.plot(time, traj[:, i], label=label)
            ax.set_xlabel('time')
            ax.set_ylabel(coord_names[i])
            ax.set_title(f'{coord_names[i]}')
            if labels is not None:
                ax.legend()
            ax.grid(True, alpha=0.3)
        
        # График ошибки от baseline
        if baseline is not None:
            ax = axes[n_dims]
            error = np.linalg.norm(traj - baseline, axis=1)
            ax.plot(time, error, linewidth=2, label=label)
            ax.set_xlabel('time')
            ax.set_ylabel('RMSE')
            ax.set_title('Delta from baseline')
            if labels is not None:
                ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Скрываем лишние подграфики, если есть
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_trajectory_comparison_nd(true_traj, pred_traj, time, coord_names=None):
    """
    Сравнение истинной и предсказанной траекторий произвольной размерности
    
    Parameters:
    -----------
    true_traj : np.array
        Истинная траектория shape (n_points, n_dims)
    pred_traj : np.array
        Предсказанная траектория shape (n_points, n_dims)
    time : np.array
        Временные метки shape (n_points,)
    coord_names : list of str, optional
        Список названий координат, если None, используются ['X', 'Y', 'Z', 'W', 'V', 'U', 'Dim_0'...]
    """
    # Определяем размерность
    n_dims = true_traj.shape[1]
    
    # Создаем сетку графиков: n_dims координат + 1 для ошибки
    n_plots = n_dims + 1
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axes = np.array(axes).flatten()  # Приводим к плоскому массиву
    
    # Названия координат
    if coord_names is None:
        coord_names = ['X', 'Y', 'Z', 'W', 'V', 'U']
        if n_dims > len(coord_names):
            coord_names.extend([f'Dim_{i}' for i in range(len(coord_names), n_dims)])
    
    # Временные ряды для каждой координаты
    for i in range(n_dims):
        ax = axes[i]
        ax.plot(time, true_traj[:, i], label='true', linewidth=2)
        ax.plot(time, pred_traj[:, i], label='predicted', linestyle='--', linewidth=2)
        ax.set_xlabel('time')
        ax.set_ylabel(coord_names[i])
        ax.set_title(f'{coord_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # График ошибки предсказания
    ax = axes[n_dims]
    error = np.linalg.norm(true_traj - pred_traj, axis=1)
    ax.plot(time, error, linewidth=2)
    ax.set_xlabel('time')
    ax.set_ylabel('Error')
    ax.set_title('Prediction Error (L2 norm)')
    ax.grid(True, alpha=0.3)
    
    # Скрываем лишние подграфики, если есть
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def test_model(
    model, 
    attractor,
    initial_state=None,
    known_interval=(0, 300), 
    pred_interval=(300, 600), 
    dt=0.005,
    attractor_params=None,
    noise = 0
):
    
    # Создаем экземпляр аттрактора, если передан класс
    if isinstance(attractor, type):
        if attractor_params is None:
            attractor_params = {}
        system = attractor(**attractor_params)
    else:
        system = attractor
    
    if initial_state is None:
        initial_state = np.ones(system.n_dim)
    
    # Генерация известной части траектории
    initial_traj = system.generate_trajectory(
        initial_state=initial_state, 
        step_range=(0, known_interval[1]),
        dt=dt
    )[known_interval[0]:]
    
    if noise > 0:
        signal_std = np.std(initial_traj, axis=0)
        noise_std = signal_std * noise
        gaussian_noise = np.random.normal(0, noise_std, initial_traj.shape)
        initial_traj = initial_traj + gaussian_noise

    # Предсказание модели
    num_pred_steps = pred_interval[1] - known_interval[1]
    pred_traj = model.predict_trajectory(
        initial_sequence=initial_traj, 
        num_steps=num_pred_steps
    )[-(pred_interval[1] - pred_interval[0]):]
    
    # Истинная траектория для сравнения
    true_traj = system.generate_trajectory(
        initial_state=initial_state, 
        step_range=(0, pred_interval[1]),
        dt=dt
    )[pred_interval[0]:]
    
    # Временная ось
    time = np.arange(pred_interval[0], pred_interval[1]) * dt
    
    # Визуализация
    plot_trajectory_comparison_nd(
        true_traj=true_traj, 
        pred_traj=pred_traj, 
        time=time
    )
    
    # Вычисление метрик
    error = np.linalg.norm(true_traj - pred_traj, axis=1)
    metrics = {
        'mean_error': np.mean(error),
        'max_error': np.max(error),
        'rmse': np.sqrt(np.mean(error**2)),
        'final_error': error[-1]
    }
    
    # Вывод метрик
    print("\n" + "="*50)
    print(f"Prediction Metrics ({system.__class__.__name__})")
    print("="*50)
    print(f"Mean Error:  {metrics['mean_error']:.4f}")
    print(f"Max Error:   {metrics['max_error']:.4f}")
    print(f"RMSE:        {metrics['rmse']:.4f}")
    print(f"Final Error: {metrics['final_error']:.4f}")
    print("="*50 + "\n")
    
    return metrics
