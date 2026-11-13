"""Create (and optionally execute) notebooks/housing_exploration.ipynb.

Usage:
    python scripts/build_notebook.py            # build + execute
    python scripts/build_notebook.py --no-exec  # build only
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "housing_exploration.ipynb"

cells = [
    nbf.v4.new_markdown_cell("# HomeWise AI — Housing exploration\n\nExploratory analysis of the Kaggle *House Prices: "
                             "Advanced Regression Techniques* training data (Ames, Iowa, 2006–2010). All figures are "
                             "computed from `data/raw/train.csv`. Associations here are not causal."),
    nbf.v4.new_code_cell("import sys, pathlib\nROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()\n"
                         "sys.path.insert(0, str(ROOT))\nimport numpy as np, pandas as pd\nimport matplotlib.pyplot as plt, seaborn as sns\n"
                         "from src.data_loader import load_train, load_test\nfrom src.data_validation import validate_train_test, missing_summary\n"
                         "from src.eda import dataset_overview, target_statistics, numeric_correlations, automatic_insights, detect_outliers\n"
                         "sns.set_theme(style='whitegrid')\ntrain, test = load_train(), load_test()\ntrain.shape, test.shape"),
    nbf.v4.new_markdown_cell("## Validation"),
    nbf.v4.new_code_cell("rep = validate_train_test(train, test)\n{k: rep[k] for k in ['schema_match', 'id_overlap', 'ok']}, rep['train']['errors'], rep['train']['warnings']"),
    nbf.v4.new_code_cell("dataset_overview(train, test)"),
    nbf.v4.new_markdown_cell("## Target variable"),
    nbf.v4.new_code_cell("pd.Series(target_statistics(train['SalePrice'])).round(2)"),
    nbf.v4.new_code_cell("fig, ax = plt.subplots(1, 2, figsize=(12, 4))\nsns.histplot(train['SalePrice'], bins=50, ax=ax[0])\n"
                         "sns.histplot(np.log1p(train['SalePrice']), bins=50, ax=ax[1])\nax[0].set_title('SalePrice'); ax[1].set_title('log1p(SalePrice)');"),
    nbf.v4.new_markdown_cell("## Missing values"),
    nbf.v4.new_code_cell("missing_summary(train.drop(columns=['SalePrice'])).head(20)"),
    nbf.v4.new_markdown_cell("## Correlations with SalePrice"),
    nbf.v4.new_code_cell("numeric_correlations(train).head(15).round(3)"),
    nbf.v4.new_code_cell("top = numeric_correlations(train).head(10).index.tolist() + ['SalePrice']\nplt.figure(figsize=(9, 7))\n"
                         "sns.heatmap(train[top].corr(), annot=True, fmt='.2f', cmap='vlag', center=0);"),
    nbf.v4.new_markdown_cell("## Key relationships"),
    nbf.v4.new_code_cell("fig, ax = plt.subplots(1, 3, figsize=(16, 4))\nsns.scatterplot(data=train, x='GrLivArea', y='SalePrice', s=12, ax=ax[0])\n"
                         "sns.boxplot(data=train, x='OverallQual', y='SalePrice', ax=ax[1])\nsns.scatterplot(data=train, x='YearBuilt', y='SalePrice', s=12, ax=ax[2]);"),
    nbf.v4.new_code_cell("order = train.groupby('Neighborhood')['SalePrice'].median().sort_values().index\nplt.figure(figsize=(14, 4))\n"
                         "sns.boxplot(data=train, x='Neighborhood', y='SalePrice', order=order)\nplt.xticks(rotation=70);"),
    nbf.v4.new_markdown_cell("## Outliers"),
    nbf.v4.new_code_cell("detect_outliers(train)['large_low_price_rows']"),
    nbf.v4.new_markdown_cell("## Automatic insights (computed, not hard-coded)"),
    nbf.v4.new_code_cell("for i in automatic_insights(train):\n    print('•', i['question'], '\\n ', i['answer'], '\\n')"),
    nbf.v4.new_markdown_cell("## Model results\n\nTrained with `python scripts/train_all.py`; results are read from `outputs/reports`."),
    nbf.v4.new_code_cell("pd.read_csv(ROOT / 'outputs/reports/model_comparison.csv')[['Model', 'CV RMSLE', 'Val RMSE', 'Val R2', 'Val MAE']].round(4)"),
]


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NB.parent.mkdir(parents=True, exist_ok=True)
    if "--no-exec" not in sys.argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbf.write(nb, NB)
    print(f"Wrote {NB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
