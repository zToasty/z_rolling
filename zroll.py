import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def calculate_rolling_zscore(data_series, window=5):
    """
    Вычисляет динамический Z-score в скользящем окне
    
    Parameters:
    -----------
    data_series : pd.Series
        Временной ряд для анализа
    window : int
        Размер скользящего окна
        
    Returns:
    --------
    tuple: (z_score, rolling_mean, rolling_std)
    """
    rolling_mean = data_series.rolling(window=window).mean()
    rolling_std = data_series.rolling(window=window).std()
    z_score = (data_series - rolling_mean) / rolling_std
    
    return z_score, rolling_mean, rolling_std

def generate_sample_data(n_points=100):
    """
    Генерирует пример временного ряда для демонстрации
    """
    dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
    # Создаем данные с трендом и сезонностью
    trend = np.linspace(0, 10, n_points)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_points) / 30)
    noise = np.random.normal(0, 1, n_points)
    data = 50 + trend + seasonal + noise
    
    return pd.DataFrame({'value': data}, index=dates)

def load_data(input_file=None):
    """
    Загружает данные из файла или генерирует тестовые данные
    
    Parameters:
    -----------
    input_file : str, optional
        Путь к CSV файлу
        
    Returns:
    --------
    pd.DataFrame
        Загруженные данные
    """
    if input_file:
        if not os.path.exists(input_file):
            print(f"Ошибка: Файл {input_file} не найден")
            sys.exit(1)
        
        try:
            df = pd.read_csv(input_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            sys.exit(1)
    else:
        print("Используются тестовые данные...")
        return generate_sample_data()

def validate_data_column(df, data_column):
    """
    Проверяет наличие колонки с данными в DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Данные для проверки
    data_column : str
        Название колонки для проверки
    """
    if data_column not in df.columns:
        print(f"Ошибка: Колонка '{data_column}' не найдена в данных")
        print(f"Доступные колонки: {list(df.columns)}")
        sys.exit(1)

def print_zscore_statistics(zscore_series):
    """
    Выводит статистику по Z-score
    
    Parameters:
    -----------
    zscore_series : pd.Series
        Series с значениями Z-score
    """
    print("\nСтатистика rolling_z_score:")
    print(f"Среднее: {zscore_series.mean():.4f}")
    print(f"Стандартное отклонение: {zscore_series.std():.4f}")
    print(f"Min: {zscore_series.min():.4f}")
    print(f"Max: {zscore_series.max():.4f}")
    print(f"Количество NaN значений: {zscore_series.isna().sum()}")

def create_zscore_plots(df, data_column, window, plot_filename):
    """
    Создает визуализацию Z-score и исходных данных
    
    Parameters:
    -----------
    df : pd.DataFrame
        Данные для визуализации
    data_column : str
        Колонка с исходными данными
    window : int
        Размер скользящего окна
    plot_filename : str
        Имя файла для сохранения графика
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # График исходных данных и скользящего среднего
    axes[0].plot(df.index, df[data_column], label='Исходные данные', linewidth=1, alpha=0.7)
    axes[0].plot(df.index, df['rolling_mean'], label=f'Rolling Mean (window={window})', linewidth=2)
    axes[0].set_title('Исходный временной ряд и скользящее среднее')
    axes[0].set_ylabel('Значение')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График Z-score
    axes[1].plot(df.index, df['rolling_z_score'], label='Rolling Z-score', color='red', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Нулевая линия')
    axes[1].axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='±2 стандартных отклонения')
    axes[1].axhline(y=-2, color='orange', linestyle='--', alpha=0.7)
    axes[1].set_title(f'Динамический Z-score (окно={window})')
    axes[1].set_ylabel('Z-score')
    axes[1].set_xlabel('Дата')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"График сохранен как: {plot_filename}")

def process_rolling_zscore(args):
    """
    Основная функция обработки rolling z-score
    
    Parameters:
    -----------
    args : argparse.Namespace
        Аргументы командной строки
    """
    # Загрузка данных
    df = load_data(args.input_file)
    
    # Проверка наличия колонки с данными
    validate_data_column(df, args.data_column)
    
    print(f"Размер окна: {args.window}")
    print(f"Количество точек данных: {len(df)}")
    print(f"Колонка с данными: '{args.data_column}'")
    
    # Вычисление rolling z-score
    df['rolling_z_score'], df['rolling_mean'], df['rolling_std'] = calculate_rolling_zscore(
        df[args.data_column], args.window
    )
    
    # Статистика
    print_zscore_statistics(df['rolling_z_score'])
    
    # Формируем имена файлов
    csv_filename = f"{args.output_file}.csv"
    plot_filename = f"{args.output_file}.png"
    
    # Сохранение результатов
    df.to_csv(csv_filename)
    print(f"\nРезультаты сохранены в: {csv_filename}")
    
    # Построение графиков
    if not args.no_plot:
        create_zscore_plots(df, args.data_column, args.window, plot_filename)

def main():
    """
    Основная функция - парсинг аргументов и запуск обработки
    """
    parser = argparse.ArgumentParser(description='Создание признака rolling z-score для временных рядов')
    parser.add_argument('--window', type=int, default=5, 
                       help='Размер скользящего окна (по умолчанию: 5)')
    parser.add_argument('--input-file', type=str, 
                       help='Путь к CSV файлу с данными. Если не указан, будут сгенерированы тестовые данные')
    parser.add_argument('--output-file', type=str, default='rolling_zscore_result',
                       help='Базовое имя для файлов результатов (по умолчанию: rolling_zscore_result)')
    parser.add_argument('--data-column', type=str, default='value',
                       help='Название колонки с данными (по умолчанию: "value")')
    parser.add_argument('--no-plot', action='store_true',
                       help='Не строить графики')
    
    args = parser.parse_args()
    
    # Запуск основной логики
    process_rolling_zscore(args)

if __name__ == "__main__":
    main()