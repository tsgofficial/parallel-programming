"""
plot_results.py — 1D Дулааны тархалтын графикийг зурна
──────────────────────────────────────────────────────────
Ажиллуулах:
    python3 plot_results.py output_100_steps.csv
    python3 plot_results.py output_1000_steps.csv
    python3 plot_results.py                         # хоёулангийг нь (хэрэв байвал)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.grid': True,
    'grid.alpha': 0.4,
})


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def extract_steps(filename: str) -> str:
    """Filename-аас цагийн алхмын тоог унших."""
    base = os.path.basename(filename)
    parts = base.replace('.csv', '').split('_')
    for i, p in enumerate(parts):
        if p == 'steps' and i > 0:
            return parts[i - 1]
        if p.isdigit():
            return p
    return 'N'


def plot_single(ax, df: pd.DataFrame, steps: str, color: str):
    """Нэг CSV файлын температурын тархалтыг зурна."""
    pos  = df['position'].values
    temp = df['temperature'].values

    ax.plot(pos, temp, color=color, linewidth=1.8, label=f't = {steps} алхам')
    ax.fill_between(pos, temp, alpha=0.12, color=color)

    # Онцлох цэгүүд
    ax.axhline(100, color='red',  linestyle='--', linewidth=0.8, alpha=0.6, label='Зүүн BC (100°)')
    ax.axhline(0,   color='blue', linestyle='--', linewidth=0.8, alpha=0.6, label='Баруун BC (0°)')

    mid_temp = temp[len(temp) // 2]
    ax.annotate(
        f'Дунд: {mid_temp:.2f}°',
        xy=(len(temp) // 2, mid_temp),
        xytext=(len(temp) // 2 + 200, mid_temp + 5),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='dimgray'
    )

    ax.set_xlabel('Байрлал (position)', fontsize=11)
    ax.set_ylabel('Температур (°)', fontsize=11)
    ax.set_title(f'1D Дулааны Тархалт — {steps} цагийн алхам', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, len(pos) - 1)
    ax.set_ylim(-5, 110)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else []

    # Аргумент байхгүй бол default файлуудыг хайна
    if not files:
        candidates = ['output_100_steps.csv', 'output_10000_steps.csv', 'output_1000000_steps.csv']
        files = [f for f in candidates if os.path.exists(f)]
        if not files:
            print("❌  CSV файл олдсонгүй. Жишээ: python3 plot_results.py output_100_steps.csv")
            sys.exit(1)

    print(f"📊  {len(files)} файл олдлоо: {', '.join(files)}")

    colors = ['#e63946', '#2196f3', '#4caf50', '#ff9800']

    if len(files) == 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        df = load_csv(files[0])
        steps = extract_steps(files[0])
        plot_single(ax, df, steps, colors[0])
        out = files[0].replace('.csv', '.png')

    else:
        # Харьцуулалтын график: олон файл нэг зурагт
        n = len(files)
        fig = plt.figure(figsize=(12, 5 * n))
        gs  = GridSpec(n + 1, 1, figure=fig, hspace=0.45)

        datasets = []
        for i, f in enumerate(files):
            ax = fig.add_subplot(gs[i])
            df = load_csv(f)
            steps = extract_steps(f)
            datasets.append((df, steps))
            plot_single(ax, df, steps, colors[i % len(colors)])

        # Хамтарсан харьцуулалтын панел
        ax_cmp = fig.add_subplot(gs[n])
        for i, (df, steps) in enumerate(datasets):
            ax_cmp.plot(
                df['position'].values,
                df['temperature'].values,
                linewidth=1.5,
                color=colors[i % len(colors)],
                label=f't = {steps} алхам'
            )
        ax_cmp.set_xlabel('Байрлал', fontsize=11)
        ax_cmp.set_ylabel('Температур (°)', fontsize=11)
        ax_cmp.set_title('Харьцуулалт — Цагийн алхмаар', fontsize=13, fontweight='bold')
        ax_cmp.legend()
        ax_cmp.set_xlim(0, 4095)
        ax_cmp.set_ylim(-5, 110)

        out = 'heat_diffusion_comparison.png'

    plt.suptitle(
        '1D Heat Diffusion — Finite Difference Method (MPI)\n'
        'N=4096, C=0.25, Rank 0 left BC=100°',
        fontsize=11, color='gray', y=1.01
    )

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅  График хадгалагдлаа: {out}")
    plt.show()


if __name__ == '__main__':
    main()
