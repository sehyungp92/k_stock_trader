"""Static research figures from completed, registered portfolio replay results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_portfolio_allocation_study import case_paths

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    root = args.study.parent
    study = read(args.study)
    final = read(root / 'finalist.json')
    results = final['results']
    winner = final['winner']
    paths = case_paths(args.study)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(left=.065,right=.985,bottom=.20,top=.88,wspace=.15)
    references = ('kalcb_native', 'olr_native', 'kalcb_size_125', 'kalcb_size_150', 'kalcb_size_175', 'kalcb_size_200')
    short = {'kalcb_native': 'KALCB', 'olr_native': 'OLR', 'kalcb_size_125': 'KALCB 1.25',
             'kalcb_size_150': 'KALCB 1.50', 'legacy_selected': 'Previous portfolio'}
    short.update(kalcb_size_175='KALCB 1.75', kalcb_size_200='KALCB 2.00')
    for i, ax in enumerate(axes):
        limit = final['selection']['baseline_max_dd_pct' if i == 0 else 'combined_max_dd_pct']
        ax.axvline(limit, color='#b64d42', linestyle='--', linewidth=1, label='Registered DD gate')
        for name, pair in results.items():
            result = pair[i]
            color = '#c47b25' if name in references else '#8796a2'
            marker = 'D' if name in references else 'o'
            if name == winner:
                color, marker = '#087f6b', '*'
            area = 270 if name == winner else 40 + result['trades'] / 6
            ax.scatter(result['drawdown_pct'], result['return_pct'], s=area,
                       c=color, marker=marker, edgecolors='white', linewidths=.6, zorder=3)
            if name in short or name == winner:
                label = 'Selected portfolio' if name == winner else short[name]
                offset=(5,7)
                if name=='kalcb_size_175':offset=(-15,-17)
                if name=='legacy_selected':offset=(5,-17)
                ax.annotate(label, (result['drawdown_pct'], result['return_pct']),
                            xytext=offset, textcoords='offset points', fontsize=8,zorder=5)
        ax.set(title='Baseline' if i == 0 else 'Combined cost / participation stress',
               xlabel='Marked maximum drawdown (%)', ylabel='Net return (%)')
        ax.grid(alpha=.18)
        ax.margins(x=.22, y=.20)
    fig.suptitle('Native portfolio allocation study | KRW100m | 156 sessions', fontsize=15)
    fig.text(.5, .025, 'Grey: portfolio recipes. Orange diamonds: standalone sizing. Bubble area increases with trade count.\n'
             'Dashed lines: registered drawdown gates. Green star: selected portfolio.\n'
             'Reused development data, 7 Aug 2025–31 Mar 2026. Historical DD gates do not guarantee future risk.',
             ha='center', fontsize=9)
    fig.savefig(root / 'allocation_frontier.png', dpi=180, bbox_inches='tight')
    fig.savefig(root / 'allocation_frontier.svg', bbox_inches='tight')
    plt.close(fig)
    if winner:
        reference = max(final['reference_frontier'], key=lambda name: results[name][1]['return_pct'])
        books = list(dict.fromkeys([winner, 'legacy_selected', reference, 'olr_native']))
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        colors = ['#087f6b', '#8796a2', '#c47b25', '#7963a0']
        for name, color in zip(books, colors):
            path = paths[name + '_baseline'] / 'daily.json'
            daily = read(path)
            values = [study['initial_equity'], *[x['equity'] for x in daily]]
            ax.plot(range(len(values)), [100*x/study['initial_equity'] for x in values],
                    label=('Selected portfolio' if name == winner else short.get(name, name)),
                    linewidth=2 if name == winner else 1.4, color=color)
        ax.set(title='Baseline marked account equity', xlabel='Exchange sessions since 7 Aug 2025',
               ylabel='Starting equity = 100')
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
        fig.savefig(root / 'selected_equity_comparison.png', dpi=180)
        plt.close(fig)
    print(json.dumps({'frontier': str(root / 'allocation_frontier.png'), 'winner': winner}))


if __name__ == '__main__':
    main()
