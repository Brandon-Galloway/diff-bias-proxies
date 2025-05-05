import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

results_dir = './results/logs'
vis_dir = './results/visualizations'
os.makedirs(vis_dir, exist_ok=True)

valid_files = [f for f in os.listdir(results_dir) if f.startswith('mimic_cxr_sex_eod_valid_output')]
test_files = [f for f in os.listdir(results_dir) if f.startswith('mimic_cxr_sex_eod_test_output')]

def load_metrics(files):
    records = []
    for fname in files:
        path = os.path.join(results_dir, fname)
        with open(path, 'r') as f:
            data = json.load(f)
        seed = int(fname.split('_')[-1].replace('.json', ''))
        for method, value in data.items():
            if method == "config":
                continue
            if isinstance(value, dict) and value:
                first = list(value.values())[0]
                if isinstance(first, dict) and all(k in first for k in ['performance', 'bias']):
                    records.append({
                        'seed': seed,
                        'method': method,
                        'performance': first['performance'],
                        'bias': first['bias']
                    })
    return pd.DataFrame(records)

def draw_bar_labels(ax, bars, values):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
            color='black',
            path_effects=[
                path_effects.withStroke(linewidth=2, foreground='white')
            ]
        )

def plot_metric(df, metric, title, fname_prefix):
    grouped = df.groupby('method')[metric].agg(['mean', 'std']).sort_values(by='mean', ascending=False)
    means = grouped['mean']
    stds = grouped['std']
    methods_sorted = means.index.tolist()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods_sorted, means.values, yerr=stds.values, capsize=5)
    draw_bar_labels(plt.gca(), bars, means)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric.capitalize())
    plt.title(title)

    if metric == 'bias':
        min_val = (means - stds).min()
        max_val = (means + stds).max()
    else:
        min_val = means.min()
        max_val = means.max()

    if max_val != min_val:
        padding = (max_val - min_val) * 0.02
        plt.ylim(min_val - padding, max_val + padding)
    else:
        plt.ylim(min_val - 0.01, max_val + 0.01)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(vis_dir, f'{fname_prefix}_{metric}.{ext}'))
    plt.close()

def plot_grouped_metrics(df, title, fname_prefix):
    grouped = df.groupby('method').agg({
        'performance': ['mean', 'std'],
        'bias': ['mean', 'std']
    })

    perf_means = grouped[('performance', 'mean')].sort_values(ascending=False)
    bias_means = grouped[('bias', 'mean')].sort_values(ascending=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for i, (metric, sorted_means) in enumerate(zip(['performance', 'bias'], [perf_means, bias_means])):
        stds = grouped[(metric, 'std')].loc[sorted_means.index]
        bars = axs[i].bar(sorted_means.index, sorted_means.values, yerr=stds.values, capsize=5)
        draw_bar_labels(axs[i], bars, sorted_means.values)
        axs[i].set_title(metric.capitalize())
        axs[i].tick_params(axis='x', rotation=45)

        if metric == 'bias':
            min_val = (sorted_means - stds).min()
            max_val = (sorted_means + stds).max()
        else:
            min_val = sorted_means.min()
            max_val = sorted_means.max()

        if max_val != min_val:
            padding = (max_val - min_val) * 0.02
            axs[i].set_ylim([min_val - padding, max_val + padding])
        else:
            axs[i].set_ylim([min_val - 0.01, max_val + 0.01])

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(vis_dir, f'{fname_prefix}_bias_perf.{ext}'))
    plt.close()

def plot_per_seed(df, split_name):
    for seed in sorted(df['seed'].unique()):
        seed_df = df[df['seed'] == seed]
        seed_dir = os.path.join(vis_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        for i, metric in enumerate(['performance', 'bias']):
            sorted_df = seed_df.sort_values(metric, ascending=False)
            bars = axs[i].bar(sorted_df['method'], sorted_df[metric])
            draw_bar_labels(axs[i], bars, sorted_df[metric].values)
            axs[i].set_title(f'{metric.capitalize()}')
            axs[i].tick_params(axis='x', rotation=45)

            min_val = sorted_df[metric].min()
            max_val = sorted_df[metric].max()
            if max_val != min_val:
                padding = (max_val - min_val) * 0.02
                axs[i].set_ylim(min_val - padding, max_val + padding)
            else:
                axs[i].set_ylim(min_val - 0.01, max_val + 0.01)

        fig.suptitle(f'{split_name.capitalize()} Results - Seed {seed}')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(seed_dir, f'{split_name}_seed_{seed}.{ext}'))
        plt.close()

valid_df = load_metrics(valid_files)
test_df = load_metrics(test_files)

plot_metric(valid_df, 'performance', 'Validation Performance by Method', 'valid')
plot_metric(valid_df, 'bias', 'Validation Bias by Method', 'valid')

plot_metric(test_df, 'performance', 'Test Performance by Method', 'test')
plot_metric(test_df, 'bias', 'Test Bias by Method', 'test')

plot_grouped_metrics(valid_df, 'Validation Bias & Performance by Method', 'valid')
plot_grouped_metrics(test_df, 'Test Bias & Performance by Method', 'test')

plot_per_seed(valid_df, 'valid')
plot_per_seed(test_df, 'test')
