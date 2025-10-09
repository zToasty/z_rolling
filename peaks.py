import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

from zroll import calculate_rolling_zscore, load_data, validate_data_column


def detect_peaks(data_series, window=5, threshold=2.0):
    """
    Обнаруживает пики во временном ряду на основе rolling z-score
    
    Parameters:
    -----------
    data_series : pd.Series
        Временной ряд для анализа
    window : int
        Размер скользящего окна
    threshold : float
        Порог Z-score для определения пика
        
    Returns:
    --------
    tuple: (peak_indicator, peak_count, z_scores)
        peak_indicator: бинарный признак пика (1 - пик, 0 - не пик)
        peak_count: общее количество пиков
        z_scores: значения Z-score
    """
    # Вычисляем Z-score
    z_scores, rolling_mean, rolling_std = calculate_rolling_zscore(data_series, window)
    
    # Создаем бинарный признак пика (|z| > threshold)
    peak_indicator = (abs(z_scores) > threshold).astype(int)
    
    # Подсчитываем количество пиков (исключая NaN значения)
    peak_count = peak_indicator.sum()
    
    return peak_indicator, peak_count, z_scores


def print_statistics(df, total_peaks):
    """
    Выводит статистику по обнаруженным пикам
    
    Parameters:
    -----------
    df : pd.DataFrame
        Данные с результатами
    total_peaks : int
        Общее количество пиков
    """
    print(f"\nОбщее количество пиков: {total_peaks}")
    print(f"Доля пиков в данных: {total_peaks/len(df):.4f}")
    

def create_plots(df, data_column, total_peaks, threshold, window, plot_filename):
    """
    Создает визуализацию результатов с учетом abs(z) > threshold
    
    Parameters:
    -----------
    df : pd.DataFrame
        Данные для визуализации
    data_column : str
        Колонка с исходными данными
    total_peaks : int
        Количество пиков
    threshold : float
        Порог Z-score
    window : int
        Размер окна
    plot_filename : str
        Имя файла для сохранения графика
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # График 1: Исходные данные с выделенными пиками
    axes[0].plot(df.index, df[data_column], label='Исходные данные', linewidth=1, alpha=0.7, color='blue')
    
    # Выделяем пики красными точками (и положительные, и отрицательные)
    peak_points = df[df['is_peak'] == 1]
    axes[0].scatter(peak_points.index, peak_points[data_column], 
                   color='red', s=50, zorder=5, label=f'Пики (|z| > {threshold})')
    
    axes[0].set_title(f'Обнаружение пиков во временном ряду (найдено: {total_peaks})')
    axes[0].set_ylabel('Значение')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График 2: Z-score с порогами
    axes[1].plot(df.index, df['z_score'], label='Z-score', color='green', linewidth=1.5)
    
    # Верхний порог
    axes[1].axhline(y=threshold, color='red', linestyle='--', alpha=0.8, 
                   label=f'Порог пика (|z| = {threshold})')
    # Нижний порог
    axes[1].axhline(y=-threshold, color='red', linestyle='--', alpha=0.8)
    # Нулевая линия
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Нулевая линия')
    
    # Закрашиваем обе области пиков
    axes[1].fill_between(df.index, threshold, df['z_score'], 
                        where=(df['z_score'] > threshold), 
                        color='red', alpha=0.3, label='Область пиков')
    axes[1].fill_between(df.index, -threshold, df['z_score'], 
                        where=(df['z_score'] < -threshold), 
                        color='red', alpha=0.3)
    
    axes[1].set_title(f'Z-score и порог обнаружения пиков (окно: {window})')
    axes[1].set_ylabel('Z-score')
    axes[1].set_xlabel('Дата')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Добавляем информацию о распределении пиков
    positive_peaks = (df['z_score'] > threshold).sum()
    negative_peaks = (df['z_score'] < -threshold).sum()
    
    # Текст с информацией о пиках
    peak_info = f'Положительные пики (z > {threshold}): {positive_peaks}\nОтрицательные пики (z < -{threshold}): {negative_peaks}'
    axes[1].text(0.02, 0.98, peak_info, transform=axes[1].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"График сохранен как: {plot_filename}")
    print(f"Распределение пиков: положительные - {positive_peaks}, отрицательные - {negative_peaks}")

def process_peaks_detection(args):
    """
    Основная функция обработки обнаружения пиков
    
    Parameters:
    -----------
    args : argparse.Namespace
        Аргументы командной строки
    """
    # Загрузка и валидация данных
    df = load_data(args.input_file)
    validate_data_column(df, args.data_column)
    
    print(f"Размер окна: {args.window}")
    print(f"Порог Z-score: {args.threshold}")
    print(f"Количество точек данных: {len(df)}")
    print(f"Колонка с данными: '{args.data_column}'")
    
    # Обнаружение пиков
    df['is_peak'], total_peaks, z_scores = detect_peaks(df[args.data_column], args.window, args.threshold)
    
    # Добавляем z-score в DataFrame для полноты
    df['z_score'] = z_scores
    
    # Вывод статистики
    print_statistics(df, total_peaks)
    
    # Формируем имена файлов
    csv_filename = f"{args.output_file}.csv"
    plot_filename = f"{args.output_file}.png"
    
    # Сохранение результатов
    df.to_csv(csv_filename)
    print(f"\nРезультаты сохранены в: {csv_filename}")
    
    # Построение графиков
    if not args.no_plot:
        create_plots(df, args.data_column, total_peaks, args.threshold, args.window, plot_filename)


def main():
    """
    Основная функция - парсинг аргументов и запуск обработки
    """
    parser = argparse.ArgumentParser(description='Обнаружение пиков во временном ряду на основе rolling z-score')
    parser.add_argument('--window', type=int, default=5, 
                       help='Размер скользящего окна (по умолчанию: 5)')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Порог Z-score для определения пика (по умолчанию: 2.0)')
    parser.add_argument('--input-file', type=str, 
                       help='Путь к CSV файлу с данными. Если не указан, будут сгенерированы тестовые данные')
    parser.add_argument('--output-file', type=str, default='peaks_detection_result.csv',
                       help='Путь для сохранения результатов (по умолчанию: peaks_detection_result.csv)')
    parser.add_argument('--data-column', type=str, default='value',
                       help='Название колонки с данными (по умолчанию: "value")')
    parser.add_argument('--no-plot', action='store_true',
                       help='Не строить графики')
    
    args = parser.parse_args()
    
    process_peaks_detection(args)


if __name__ == "__main__":
    main()