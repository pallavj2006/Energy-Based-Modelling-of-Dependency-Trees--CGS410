import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import random
import warnings
import os
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

CONLLU_FILES = {
    'English':  'en_ewt-ud-train.conllu',
    'Hindi':    'hi_hdtb-ud-train.conllu',
    'German':   'de_gsd-ud-train.conllu',
    'Spanish':  'es_gsd-ud-train.conllu',
    'Japanese': 'ja_gsd-ud-train.conllu',
    'Turkish':  'tr_boun-ud-train.conllu',
    'Chinese':  'zh_gsd-ud-train.conllu',
    'Arabic':   'ar_padt-ud-train.conllu',
}


def parse_conllu(filepath):
    sentences = []
    current = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                continue
            if line == '':
                if current:
                    sentences.append(current)
                    current = []
                continue
            fields = line.split('\t')
            if len(fields) < 8:
                continue
            token_id = fields[0]
            if '-' in token_id or '.' in token_id:
                continue
            try:
                tid  = int(token_id)
                head = int(fields[6])
            except ValueError:
                continue
            current.append((tid, head))
    if current:
        sentences.append(current)
    return sentences


def compute_metrics(sentences):
    rows = []
    for sent in sentences:
        n = len(sent)
        if n < 3 or n > 30:
            continue

        tids      = {tok[0] for tok in sent}
        heads_map = {tok[0]: tok[1] for tok in sent}

        if any(h != 0 and h not in tids for h in heads_map.values()):
            continue

        is_head_tid = set()
        for tid, head in sent:
            if head != 0:
                is_head_tid.add(head)

        dl_list, ic_list = [], []
        for tid, head in sent:
            if head == 0:
                continue
            dl = abs(head - tid)
            dl_list.append(dl)
            lo, hi = min(tid, head), max(tid, head)
            ic = sum(1 for h in is_head_tid if lo < h < hi)
            ic_list.append(ic)

        if not dl_list:
            continue

        real_dl = np.mean(dl_list)
        real_ic = np.mean(ic_list)

        # random baseline: shuffle word positions, recompute DL and IC on same arc pairs
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        new_pos = {tok[0]: perm[i] for i, tok in enumerate(sent)}

        rand_is_head_pos = {new_pos[h] for h in is_head_tid if h in new_pos}

        rand_dl_list, rand_ic_list = [], []
        for tid, head in sent:
            if head == 0:
                continue
            dp = new_pos[tid]
            hp = new_pos[head]
            rand_dl_list.append(abs(hp - dp))
            lo_r, hi_r = min(dp, hp), max(dp, hp)
            ic_r = sum(1 for h in rand_is_head_pos if lo_r < h < hi_r)
            rand_ic_list.append(ic_r)

        rows.append({
            'length':   n,
            'real_dl':  real_dl,
            'real_ic':  real_ic,
            'rand_dl':  np.mean(rand_dl_list),
            'rand_ic':  np.mean(rand_ic_list),
        })

    return pd.DataFrame(rows)


all_data   = {}
lang_stats = {}

for lang, path in CONLLU_FILES.items():
    sents = parse_conllu(path)
    df    = compute_metrics(sents)
    all_data[lang] = df

    rdl  = df['real_dl'].mean()
    rndl = df['rand_dl'].mean()
    ric  = df['real_ic'].mean()
    rnic = df['rand_ic'].mean()
    pct  = (rndl - rdl) / rndl * 100
    u_dl, p_dl = mannwhitneyu(df['real_dl'], df['rand_dl'], alternative='less')
    u_ic, p_ic = mannwhitneyu(df['real_ic'], df['rand_ic'], alternative='less')
    DL_MIN = 1.0
    beta = round(1.0 / (rdl - DL_MIN), 3) if rdl > DL_MIN else float('nan')

    lang_stats[lang] = {
        'n_sents':      len(df),
        'real_dl':      round(rdl, 3),
        'rand_dl':      round(rndl, 3),
        'dl_pct':       round(pct, 1),
        'p_dl':         p_dl,
        'real_ic':      round(ric, 3),
        'rand_ic':      round(rnic, 3),
        'p_ic':         p_ic,
        'beta':         beta,
        'real_dl_sem':  df['real_dl'].sem(),
        'rand_dl_sem':  df['rand_dl'].sem(),
        'real_ic_sem':  df['real_ic'].sem(),
        'rand_ic_sem':  df['rand_ic'].sem(),
    }

print(f"{'Lang':<10} {'N':>6} {'RDL':>6} {'BaseDL':>7} {'DL%':>6} "
      f"{'p_DL':>10} {'RIC':>6} {'BaseIC':>7} {'p_IC':>10} {'beta':>6}")
for lang, s in lang_stats.items():
    print(f"{lang:<10} {s['n_sents']:>6} {s['real_dl']:>6.3f} {s['rand_dl']:>7.3f} "
          f"{s['dl_pct']:>5.1f}% "
          f"{s['p_dl']:>10.2e} {s['real_ic']:>6.3f} "
          f"{s['rand_ic']:>7.3f} {s['p_ic']:>10.2e} {s['beta']:>6.3f}")

pd.DataFrame(lang_stats).T.to_csv('summary_table.csv')

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'Real': '#2C6E9B', 'Random': '#D44E3A'}
LANGS  = list(CONLLU_FILES.keys())

# Figure 1: mean DL real vs random
real_dl_m = [lang_stats[l]['real_dl']     for l in LANGS]
rand_dl_m = [lang_stats[l]['rand_dl']     for l in LANGS]
real_dl_s = [lang_stats[l]['real_dl_sem'] for l in LANGS]
rand_dl_s = [lang_stats[l]['rand_dl_sem'] for l in LANGS]
pcts      = [lang_stats[l]['dl_pct']      for l in LANGS]

x, w = np.arange(len(LANGS)), 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x-w/2, real_dl_m, w, yerr=real_dl_s, capsize=4, color=COLORS['Real'],
       alpha=0.88, label='Real trees', error_kw={'linewidth':1.2})
ax.bar(x+w/2, rand_dl_m, w, yerr=rand_dl_s, capsize=4, color=COLORS['Random'],
       alpha=0.88, label='Random baseline', error_kw={'linewidth':1.2})
ax.set_xticks(x); ax.set_xticklabels(LANGS, fontsize=11)
ax.set_ylabel('Mean Dependency Length', fontsize=12)
ax.set_title('Mean Dependency Length: Real vs Random Trees\n(sentences 3-30 tokens; error bars = SEM)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, max(rand_dl_m) * 1.25)
for i, (rm, vm, pct) in enumerate(zip(real_dl_m, rand_dl_m, pcts)):
    ax.annotate(f'-{pct:.1f}%', xy=(i, max(rm, vm)+0.12), ha='center', fontsize=9, color='#333333')
plt.tight_layout()
plt.savefig('figures/fig1_dl_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: DL vs sentence length
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()
for idx, lang in enumerate(LANGS):
    df, ax = all_data[lang], axes[idx]
    for col, label, color in [('real_dl','Real',COLORS['Real']), ('rand_dl','Random',COLORS['Random'])]:
        means, mids = [], []
        for b in range(3, 28, 3):
            ch = df[(df['length'] >= b) & (df['length'] < b+3)]
            if len(ch) > 10:
                means.append(ch[col].mean()); mids.append(b+1.5)
        ax.plot(mids, means, 'o-', color=color, label=label, linewidth=1.8, markersize=4, alpha=0.9)
    ax.set_title(lang, fontsize=11, fontweight='bold')
    ax.set_xlabel('Sentence Length', fontsize=9); ax.set_ylabel('Mean DL', fontsize=9)
    if idx == 0: ax.legend(fontsize=8)
plt.suptitle('Mean Dependency Length vs Sentence Length by Language', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/fig2_dl_by_length.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: IC real vs random
real_ic_m = [lang_stats[l]['real_ic']     for l in LANGS]
rand_ic_m = [lang_stats[l]['rand_ic']     for l in LANGS]
real_ic_s = [lang_stats[l]['real_ic_sem'] for l in LANGS]
rand_ic_s = [lang_stats[l]['rand_ic_sem'] for l in LANGS]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x-w/2, real_ic_m, w, yerr=real_ic_s, capsize=4, color=COLORS['Real'],
       alpha=0.88, label='Real trees', error_kw={'linewidth':1.2})
ax.bar(x+w/2, rand_ic_m, w, yerr=rand_ic_s, capsize=4, color=COLORS['Random'],
       alpha=0.88, label='Random baseline', error_kw={'linewidth':1.2})
ax.set_xticks(x); ax.set_xticklabels(LANGS, fontsize=11)
ax.set_ylabel('Mean Intervener Complexity', fontsize=12)
ax.set_title('Mean Intervener Complexity: Real vs Random Trees\n(error bars = SEM)', fontsize=12, fontweight='bold')
ax.legend(fontsize=11); ax.set_ylim(0, max(rand_ic_m)*1.25)
plt.tight_layout()
plt.savefig('figures/fig3_ic_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: DL reduction ranked
reductions  = sorted([(l, lang_stats[l]['dl_pct']) for l in LANGS], key=lambda x: x[1])
langs_s, vals_s = zip(*reductions)
avg_red = np.mean(vals_s)
fig, ax = plt.subplots(figsize=(8, 5))
colors_g = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(langs_s)))
bars = ax.barh(list(langs_s), list(vals_s), color=colors_g, alpha=0.9, edgecolor='white')
for bar, val in zip(bars, vals_s):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=11)
ax.set_xlabel('DL Reduction vs Random Baseline (%)', fontsize=12)
ax.set_title('Dependency Length Minimization Across Languages\n(% reduction: real vs random)',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, max(vals_s)*1.3)
ax.axvline(x=avg_red, color='#333333', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(avg_red+0.4, -0.5, f'Avg: {avg_red:.1f}%', fontsize=9, color='#333333')
plt.tight_layout()
plt.savefig('figures/fig4_dl_reduction.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 5: energy model illustration
dl_vals = np.linspace(0.5, 10, 300)
fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.cm.plasma
for i, beta in enumerate([0.0, 0.3, 0.6, 1.0, 1.5]):
    probs = np.exp(-beta * dl_vals)
    probs /= np.trapezoid(probs, dl_vals)
    color = cmap(i / 4)
    ax.plot(dl_vals, probs, color=color, linewidth=2,
            label=f'beta = {beta}' if beta > 0 else 'beta = 0 (uniform)')
ax.set_xlabel('Dependency Length (DL)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Energy-Based Distribution: P(tree) ~ exp(-beta * DL)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.set_xlim(0.5, 10)
plt.tight_layout()
plt.savefig('figures/fig5_energy_model.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 6: beta estimates
betas_sorted = sorted([(l, lang_stats[l]['beta']) for l in LANGS], key=lambda x: x[1])
b_langs, b_vals = zip(*betas_sorted)
fig, ax = plt.subplots(figsize=(8, 5))
colors_b = plt.cm.Blues(np.linspace(0.4, 0.9, len(b_langs)))
bars = ax.barh(list(b_langs), list(b_vals), color=colors_b, edgecolor='white', alpha=0.9)
for bar, val in zip(bars, b_vals):
    ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)
ax.set_xlabel('Estimated beta (MLE under exponential energy model)', fontsize=11)
ax.set_title('Estimated Inverse-Temperature (beta) Across Languages', fontsize=11, fontweight='bold')
ax.set_xlim(0, max(b_vals)*1.2)
plt.tight_layout()
plt.savefig('figures/fig6_beta_estimates.png', dpi=150, bbox_inches='tight')
plt.close()
