"""Загрузка чеклиста лояльности v4 (пять групп, оба сегмента) из локальных артефактов.

В `loyalty_v4_all_cols.parquet` лежат **оба** сегмента витрины — исходный
`SEGMENT_OLD` (`segment=0`) и добавленный позже `SEGMENT_NEW` (`segment=1`),
в каждом по пять групп (`group_id`), каждая со своей траекторией на общей
сетке дат. `group_id` одинаковый в обоих сегментах, поэтому `to_trajectory`
(пивот по `group_id`) работает только на одном сегменте за раз и падает,
если во входном df их больше одного. `load_loyalty_df(segment=...)` делает
этот отбор: по умолчанию — `SEGMENT_OLD`, чтобы код, писавшийся до появления
второго сегмента, продолжал получать те же данные без правок;
`segment=SEGMENT_NEW` — новый; `segment=None` — оба сегмента без фильтра
(тогда `to_trajectory` вызывать нельзя, только смотреть df напрямую).
`to_trajectory` возвращает `train_data` формы `(5, n_timesteps, n_dims)` — тот
же многотраекторный формат, что у `generate_attractor_data`.

Отбор признаков здесь не делается: `load_loyalty_df` отдаёт все колонки, какие
есть в файле, а порог по максимуму доли и исключение отдельных признаков —
дело вызывающей стороны (`thesis_demo.ipynb`). Поэтому читается
`loyalty_v4_all_cols.parquet` — все 104 признака, до порога `max >= 0.03`.

Окно дат, в отличие от порога, применяется здесь: в `*_all_cols.parquet` лежат
все 732 даты, включая 2026-01-01, и `DATE_TO` обрезает их до 731.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data_loyalty_v4"

# Служебные колонки: ключи грануляции, не признаки.
ID_COLS = ["segment", "group_id", "dt"]
GROUP_COL = "group_id"
SEGMENT_COL = "segment"

# Два сегмента в витрине (см. докстринг модуля; расшифровка — в
# thesis_feature_mapping.json вне этой папки).
SEGMENT_OLD = 0
SEGMENT_NEW = 1


DATE_FROM = date(2024, 1, 1)
DATE_TO = date(2026, 1, 1)


def load_loyalty_df(date_from=DATE_FROM, date_to=DATE_TO, segment=SEGMENT_OLD):
    """Читает длинный df: строка — (сегмент, группа, дата).

    Parameters
    ----------
    date_from, date_to : datetime.date | None
        Границы окна, `date_to` не включается. `None` — без обрезки.
    segment : int | None
        `0`/`1` для отбора одного сегмента; по умолчанию `SEGMENT_OLD` —
        данные, на которых считался весь предыдущий анализ. `SEGMENT_NEW` —
        второй сегмент. `None` — не фильтровать, вернуть оба (тогда
        `to_trajectory` на результате вызвать нельзя).

    Returns
    -------
    df : pd.DataFrame
        Ключи `segment`, `group_id`, `dt` и признаки `x0..x103`.
    feature_cols : list[str]
        Имена всех признаков в файле, без какого-либо отбора.
    """
    df = pd.read_parquet(DATA_DIR / "loyalty_v4_all_cols.parquet")

    if not isinstance(df["dt"].iloc[0], date):
        df["dt"] = pd.to_datetime(df["dt"]).dt.date

    if date_from is not None:
        df = df[df["dt"] >= date_from]
    if date_to is not None:
        df = df[df["dt"] < date_to]

    if segment is not None:
        df = df[df[SEGMENT_COL] == segment]
        if df.empty:
            raise ValueError(f"сегмент {segment} не найден в loyalty_v4_all_cols.parquet")

    df = df.sort_values([SEGMENT_COL, GROUP_COL, "dt"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ID_COLS]
    return df, feature_cols


def load_group_df(df, group_id):
    """Одна группа как df с колонкой `dt` — форма, в которой ряды рисуются."""
    g = df[df[GROUP_COL] == group_id].sort_values("dt").reset_index(drop=True)
    if g.empty:
        raise ValueError(f"группы {group_id} нет в данных")
    return g


def to_trajectory(df, feature_cols, group_ids=None):
    """Переводит длинный df в форматы, ожидаемые проектом.

    `df` должен содержать ровно один сегмент — `group_id` одинаковый в обоих
    сегментах витрины, так что пивот по нему на смеси сегментов задвоил бы
    сетку дат и молча сломался бы на проверке ниже. Отфильтруйте сегмент через
    `load_loyalty_df(segment=...)` до вызова.

    Parameters
    ----------
    group_ids : list | None
        Какие группы брать и в каком порядке. `None` — все, по возрастанию.

    Returns
    -------
    X : np.ndarray, shape (n_dims, n_timesteps)
        Первая из выбранных групп — форма для `svd_smoothing` и `check_traj`.
    train_data : np.ndarray, shape (n_groups, n_timesteps, n_dims)
    group_ids : list
        Порядок групп по первой оси `train_data`.
    """
    segments = df[[SEGMENT_COL]].drop_duplicates()
    if len(segments) != 1:
        raise ValueError(
            f"to_trajectory требует один сегмент во входном df, получено "
            f"{len(segments)}: {segments.to_dict('records')} — отфильтруйте "
            "через load_loyalty_df(segment=...)"
        )

    if group_ids is None:
        group_ids = sorted(df[GROUP_COL].unique())

    dates = np.sort(df["dt"].unique())
    trajectories = []
    for gid in group_ids:
        g = load_group_df(df, gid)
        if len(g) != len(dates) or not np.array_equal(g["dt"].to_numpy(), dates):
            raise ValueError(f"у группы {gid} сетка дат отличается от общей")
        trajectories.append(g[feature_cols].to_numpy(dtype=float))

    train_data = np.stack(trajectories)
    X = train_data[0].T
    return X, train_data, list(group_ids)


if __name__ == "__main__":
    df_all, _ = load_loyalty_df(segment=None)
    available = df_all[[SEGMENT_COL]].drop_duplicates()
    print(f"сегментов в файле: {len(available)} — {available.to_dict('records')}")

    df, cols = load_loyalty_df()  # segment=SEGMENT_OLD по умолчанию
    X, train_data, group_ids = to_trajectory(df, cols)
    seg = df[[SEGMENT_COL]].drop_duplicates()
    print(f"df:         rows={len(df)} columns={len(df.columns)}")
    print(f"сегмент:    {seg.iloc[0]['segment']}")
    print(f"группы:     {group_ids}")
    print(f"даты:       {df['dt'].min()} .. {df['dt'].max()}")
    print(f"признаков:  {len(cols)}")
    print(f"X:          {X.shape}   # (n_dims, n_timesteps), группа {group_ids[0]}")
    print(f"train_data: {train_data.shape}   # (n_samples, n_timesteps, n_dims)")
    print(f"NaN={int(np.isnan(train_data).sum())}  "
          f"min={train_data.min():.6f}  max={train_data.max():.6f}")
