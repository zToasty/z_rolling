import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_rolling_zscore(data_series: pd.Series, window: int = 5) -> pd.Series:
    """
    Вычисляет динамический (rolling) Z-score в скользящем окне.

    Формула:
        Z_t = (x_t - μ_t) / σ_t
    где:
        μ_t — среднее значение в окне размера `window`,
        σ_t — стандартное отклонение в этом же окне.

    Parameters
    ----------
    data_series : pd.Series
        Временной ряд для анализа.
    window : int, optional
        Размер скользящего окна (по умолчанию 5).

    Returns
    -------
    pd.Series
        Series с рассчитанными rolling Z-score для каждой точки ряда.
    """
    rolling_mean = data_series.rolling(window=window).mean()
    rolling_std = data_series.rolling(window=window).std(ddof=0)
    z_score = (data_series - rolling_mean) / rolling_std
    return z_score


def generate_sample_data(n_points: int = 100) -> pd.DataFrame:
    """
    Генерирует синтетические временные ряды для демонстрации.

    Строится ряд с линейным трендом, синусоидальной сезонностью и шумом.

    Parameters
    ----------
    n_points : int, optional
        Количество временных точек (по умолчанию 100).

    Returns
    -------
    pd.DataFrame
        DataFrame с одной колонкой "value" и индексом в виде дат.
    """
    dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
    trend = np.linspace(0, 10, n_points)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_points) / 30)
    noise = np.random.normal(0, 1, n_points)
    data = 50 + trend + seasonal + noise
    return pd.DataFrame({'value': data}, index=dates)


def load_data(input_file: str, data_column: str) -> pd.DataFrame:
    """
    Загружает данные из CSV файла и подготавливает их к обработке.

    Если присутствует колонка 'date', она конвертируется в datetime и становится индексом.

    Parameters
    ----------
    input_file : str
        Путь к CSV файлу.
    data_column : str
        Название колонки, содержащей временной ряд.

    Returns
    -------
    pd.DataFrame
        Загруженные данные в формате pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        Если указанный файл не найден.
    ValueError
        Если отсутствует нужная колонка с данными.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Файл '{input_file}' не найден")

    df = pd.read_csv(input_file)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    if data_column not in df.columns:
        raise ValueError(f"Колонка '{data_column}' не найдена. Доступные: {list(df.columns)}")

    return df


def plot_results(df: pd.DataFrame, data_column: str, window: int, output_name: str) -> None:
    """
    Визуализирует исходный временной ряд и вычисленные Z-score.

    Строятся два графика:
    1. Исходный ряд и его rolling mean.
    2. Rolling Z-score с отметками ±2σ.

    Параметр output_name определяет имя PNG файла для сохранения графика.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица с исходными и вычисленными данными.
    data_column : str
        Название исходной колонки.
    window : int
        Размер окна скользящего среднего/Z-score.
    output_name : str
        Базовое имя для сохранения графика.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Исходные данные и скользящее среднее
    axes[0].plot(df.index, df[data_column], label='Исходные данные', linewidth=1, alpha=0.7)
    axes[0].plot(df.index, df['rolling_mean'], label=f'Rolling Mean (window={window})', linewidth=2)
    axes[0].set_title('Исходный временной ряд и скользящее среднее')
    axes[0].set_ylabel('Значение')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Rolling Z-score
    axes[1].plot(df.index, df['rolling_z_score'], color='red', linewidth=1.5, label='Rolling Z-score')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1].axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='±2σ')
    axes[1].axhline(y=-2, color='orange', linestyle='--', alpha=0.7)
    axes[1].set_title(f'Динамический Z-score (окно={window})')
    axes[1].set_ylabel('Z-score')
    axes[1].set_xlabel('Дата')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = f"{output_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"График сохранён как: {plot_filename}")


def main():
    """
    Основная функция командной утилиты.

    Парсит аргументы, загружает или генерирует данные,
    вычисляет rolling Z-score и сохраняет результаты (CSV + график).
    """
    parser = argparse.ArgumentParser(description='Создание признака rolling z-score для временных рядов')
    parser.add_argument('--window', type=int, default=5, help='Размер скользящего окна (по умолчанию: 5)')
    parser.add_argument('--input-file', type=str, help='Путь к CSV файлу с данными')
    parser.add_argument('--output-name', type=str, default='rolling_zscore_result',
                        help='Базовое имя для файлов результата (без расширения)')
    parser.add_argument('--data-column', type=str, default='value', help='Колонка с данными (по умолчанию: value)')
    parser.add_argument('--no-plot', action='store_true', help='Не строить графики')
    args = parser.parse_args()

    # Загрузка или генерация данных
    if args.input_file:
        try:
            df = load_data(args.input_file, args.data_column)
        except (FileNotFoundError, ValueError) as e:
            print(f"Ошибка: {e}")
            sys.exit(1)
    else:
        print("Используются синтетические данные...")
        df = generate_sample_data()

    print(f"\nРазмер окна: {args.window}")
    print(f"Количество точек данных: {len(df)}")
    print(f"Колонка с данными: '{args.data_column}'")

    # Вычисления
    df['rolling_mean'] = df[args.data_column].rolling(window=args.window).mean()
    df['rolling_std'] = df[args.data_column].rolling(window=args.window).std()
    df['rolling_z_score'] = calculate_rolling_zscore(df[args.data_column], args.window)

    # Статистика
    print("\nСтатистика rolling_z_score:")
    print(df['rolling_z_score'].describe())

    # Сохранение
    csv_filename = f"{args.output_name}.csv"
    df.to_csv(csv_filename)
    print(f"\nРезультаты сохранены в: {csv_filename}")

    # Визуализация
    if not args.no_plot:
        plot_results(df, args.data_column, args.window, args.output_name)


if __name__ == "__main__":
    main()