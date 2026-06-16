# Energy-Based Modelling of Dependency Trees in Natural Language
### CGS410 Course Project — Pallav Jain (Roll No. 240722), IIT Kanpur

## Overview
This project tests whether dependency length minimisation (DLM) across eight typologically diverse languages can be captured by a probabilistic energy-based model of the form:

$$P(\text{tree}) \propto \exp(-\beta \times \text{DL})$$

Real dependency trees from Universal Dependencies v2 treebanks are compared against random word-order baselines on two metrics: mean dependency length (DL) and intervener complexity (IC). A maximum-likelihood estimate of $\beta$ is derived per language under the exponential family parametrisation.

**Languages covered:** English, Hindi, German, Spanish, Japanese, Turkish, Chinese, Arabic

---

## Repository Structure
```text
.
├── analysis.py          # Full analysis: parsing, metrics, stats, all figures
├── requirements.txt
├── README.md
├── data/
│   ├── en_ewt-ud-train.conllu
│   ├── hi_hdtb-ud-train.conllu
│   ├── de_gsd-ud-train.conllu
│   ├── es_gsd-ud-train.conllu
│   ├── ja_gsd-ud-train.conllu
│   ├── tr_boun-ud-train.conllu
│   ├── zh_gsd-ud-train.conllu
│   └── ar_padt-ud-train.conllu
└── figures/             # Generated automatically on run
    ├── fig1_dl_comparison.png
    ├── fig2_dl_by_length.png
    ├── fig3_ic_comparison.png
    ├── fig4_dl_reduction.png
    ├── fig5_energy_model.png
    └── fig6_beta_estimates.png
```

---

## Data
* Treebanks are sourced from **Universal Dependencies v2**. 
* Sentences outside the 3–30 token range, multiword tokens, and empty nodes are excluded from the analysis to maintain consistency.

---

## Setup
Python 3.8+ is required. No GPU or specialized hardware is needed.

```bash
pip install -r requirements.txt
```

## Running the Analysis
To run the full pipeline, execute:

```bash
python analysis.py
```
This script will print a summary table to `stdout`, write the results to `summary_table.csv`, and automatically save all six analytical plots into the `figures/` directory.

---

## Key Results

| Language | Real DL | Random DL | Reduction | $\beta$ (MLE) |
| :--- | :---: | :---: | :---: | :---: |
| **English** | 2.738 | 5.107 | 46.4% | 0.575 |
| **Hindi** | 3.065 | 6.340 | 51.7% | 0.484 |
| **German** | 3.187 | 5.784 | 44.9% | 0.457 |
| **Spanish** | 2.841 | 6.534 | 56.5% | 0.543 |
| **Japanese** | 2.340 | 6.055 | 61.4% | 0.746 |
| **Turkish** | 2.258 | 4.223 | 46.5% | 0.795 |
| **Chinese** | 3.249 | 6.647 | 51.1% | 0.445 |
| **Arabic** | 2.639 | 5.858 | 55.0% | 0.610 |

All eight Mann-Whitney U tests yield $p < 0.000125$ (the Bonferroni-corrected threshold for $\alpha = 0.001$ across eight languages). SOV head-final languages (Turkish, Japanese) show the highest $\beta$ estimates, which strongly aligns with the energy model hypothesis.

---

## Method Summary

* **DL (Dependency Length):** Absolute difference in 1-based word positions between head and dependent, averaged over non-root arcs per sentence.
* **IC (Intervener Complexity):** Number of syntactic heads intervening strictly between a head-dependent pair, averaged over non-root arcs.
* **Baseline:** One random permutation of word positions per sentence, preserving the underlying arc structure.
* **Energy Model:** Defined as $P(t \mid \beta) = \frac{\exp(-\beta \times \text{DL}(t))}{Z(\beta)}$. The MLE closed form is:
$$\hat{\beta} = \frac{1}{\text{DL}_{\text{obs}} - \text{DL}_{\text{min}}}$$ 
where $\text{DL}_{\text{min}} = 1.0$.
* **Statistical Test:** One-tailed Mann-Whitney U ($\text{real} < \text{random}$), Bonferroni-corrected at $\alpha / 8 = 0.000125$.

---

## References

1. Futrell, R., Mahowald, K., & Gibson, E. (2015). Large-scale evidence of dependency length minimization in 37 languages. *Proceedings of the National Academy of Sciences*, 112(33).
2. Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies. *Cognition*, 68(1).
3. Gibson, E., et al. (2019). How efficiency shapes human language. *Trends in Cognitive Sciences*, 23(5).
4. Nivre, J., et al. (2016). Universal Dependencies v1. *LREC 2016*.
