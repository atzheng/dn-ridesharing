import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Disable LaTeX rendering in matplotlib (use default)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "axes.labelsize": 12,
#     "legend.fontsize": 10,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
# })

switches = [120, 300, 600, 900, 1200, 1800, 3600]
fnames = [f'output/switch={s}/results.csv' for s in switches]

# Read and concatenate all results, adding a 'switch' column
def extract_switch(fname):
    match = re.search(r'switch=(\d+)', fname)
    return int(match.group(1)) if match else None

dfs = []
for fname in fnames:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        df['switch'] = extract_switch(fname)
        dfs.append(df)
xs = pd.concat(dfs, ignore_index=True)

# Read ATE and compute mean and SE
df_ate = pd.read_csv('output/ate.csv')
df_ate['delta'] = df_ate['B'] - df_ate['A']
ate = df_ate['delta'].mean()
ate_se = df_ate['delta'].std() / np.sqrt(len(df_ate))

# Prepare summary statistics
sm = xs.copy()
sm['switch'] = sm['switch'].astype(int)
sm = sm[['switch', 'dq', 'naive']]
sm = sm.melt(id_vars=['switch'], var_name='estimator', value_name='estimate')
sm['ATE'] = ate

def group_stats(df):
    ATE = df['ATE'].iloc[0]
    est = df['estimate']
    n = len(est)
    bias_mean = np.abs((est - ATE).mean()) / np.max([ATE, 1e-8])
    bias_se = est.std() / np.sqrt(n) / np.max([ATE, 1e-8])
    sd_mean = est.std() / np.max([ATE, 1e-8])
    sd_se = np.sqrt(np.var((est - est.mean())**2) / np.sqrt(n)) / np.max([ATE, 1e-8])
    rmse_mean = np.sqrt(((est - ATE) ** 2).mean()) / np.max([ATE, 1e-8])
    rmse_se = np.sqrt(np.var((est - ATE) ** 2) / np.sqrt(n)) / np.max([ATE, 1e-8])
    mean = est.mean()
    return pd.Series({
        'bias_mean': bias_mean,
        'bias_se': bias_se,
        'sd_mean': sd_mean,
        'sd_se': sd_se,
        'rmse_mean': rmse_mean,
        'rmse_se': rmse_se,
        'mean': mean,
    })

summary = sm.groupby(['estimator', 'switch']).apply(group_stats).reset_index()
summary['estimator'] = summary['estimator'].map({'naive': 'Naive', 'dq': 'DN'}).fillna(summary['estimator'])
summary['switch'] = summary['switch'] / 60  # convert to minutes

# Prepare for plotting (melt and facet)
plot_df = summary.melt(
    id_vars=['estimator', 'switch'],
    value_vars=['bias_mean', 'bias_se', 'rmse_mean', 'rmse_se', 'sd_mean', 'sd_se'],
    var_name='var', value_name='val'
)
plot_df[['metric', 'stat']] = plot_df['var'].str.split('_', expand=True)
plot_df['metric'] = plot_df['metric'].map({'bias': 'Bias', 'rmse': 'RMSE', 'sd': 'SD'})
plot_df = plot_df.pivot_table(
    index=['estimator', 'switch', 'metric'],
    columns='stat', values='val'
).reset_index()

# Plot
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
metrics = ['Bias', 'RMSE', 'SD']
colors = {'Naive': '#1f77b4', 'DN': '#ff7f0e'}

for i, metric in enumerate(metrics):
    ax = axes[i]
    for estimator in ['Naive', 'DN']:
        data = plot_df[(plot_df['estimator'] == estimator) & (plot_df['metric'] == metric)]
        ax.errorbar(
            data['switch'], data['mean'], yerr=data['se'],
            fmt='-o', label=estimator, color=colors[estimator], capsize=3
        )
    ax.set_title(metric)
    ax.set_xlabel('Switchback Period (Minutes)')
    ax.set_ylabel('Error / ATE' if i == 0 else '')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, 0.8)
    if i == 2:
        ax.legend(title='Estimator')

plt.tight_layout()
plt.savefig('ridesharing.png', dpi=300)
plt.show() 