"""Generate notebooks/01_explore_trec.ipynb (run, then execute with nbconvert).

    python notebooks/build_01_explore_trec.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore_trec.ipynb
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(
        "# 01 · Explore the TREC Clinical Trials 2021/2022 data\n\n"
        "What the patients, labels and trials look like before any modelling. Inputs are the "
        "tables written by `scripts/build_trec_tables.py` in `<data_dir>/processed/`.\n\n"
        "Labels: **2 = eligible**, **1 = excluded** (relevant but fails a criterion), "
        "**0 = not relevant**."
    ),
    code(
        "import json\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n\n"
        "from clinreason.config import load_config\n\n"
        "pd.set_option('display.max_colwidth', 120)\n"
        "PROC = Path(load_config()['data_dir']) / 'processed'\n"
        "patients = pd.read_json(PROC / 'patients.jsonl', lines=True)\n"
        "labels = pd.read_json(PROC / 'labels.jsonl', lines=True)\n"
        "trials = pd.read_json(PROC / 'trials.jsonl', lines=True)\n"
        "stats = json.loads((PROC / 'stats.json').read_text())\n"
        "LABEL = {2: 'eligible', 1: 'excluded', 0: 'not relevant'}\n"
        "print(len(patients), 'patients |', len(labels), 'labelled pairs |', "
        "len(trials), 'judged trials')"
    ),
    md("## 1. Label balance"),
    code(
        "counts = (labels.assign(label=labels.label.map(LABEL))"
        ".groupby(['year', 'label']).size().unstack())\n"
        "counts['total'] = counts.sum(axis=1)\n"
        "counts"
    ),
    code(
        "per_patient = labels.groupby(['patient_id', 'label']).size().unstack(fill_value=0)"
        ".rename(columns=LABEL)\n"
        "fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))\n"
        "for a, year in zip(ax, (2021, 2022)):\n"
        "    sub = per_patient[per_patient.index.str.startswith(str(year))]\n"
        "    sub[['eligible', 'excluded', 'not relevant']].plot.hist(\n"
        "        bins=30, alpha=0.6, ax=a, title=f'{year}: judged trials per patient, by label')\n"
        "    a.set_xlabel('trials per patient')\n"
        "plt.tight_layout()\n"
        "per_patient.describe().round(1)"
    ),
    md("## 2. One patient and three of their judged trials"),
    code(
        "pid = '2022-1'\n"
        "print(patients.set_index('patient_id').loc[pid, 'text'])"
    ),
    code(
        "T = trials.set_index('nct_id')\n\n"
        "def show_trial(nct, chars=1500):\n"
        "    t = T.loc[nct]\n"
        "    age = (t.minimum_age_years, t.maximum_age_years)\n"
        "    print(f'{nct} | {t.brief_title}')\n"
        "    print(f'conditions: {t.conditions} | sex: {t.gender} | age (years): {age} | '\n"
        "          f'healthy volunteers: {t.healthy_volunteers} | split: {t.split_status}')\n"
        "    print('--- INCLUSION ---'); print((t.inclusion_text or '')[:chars])\n"
        "    print('--- EXCLUSION ---'); print((t.exclusion_text or '')[:chars])\n\n"
        "mine = labels[labels.patient_id == pid]\n"
        "examples = {lab: mine[mine.label == lab].nct_id.iloc[0] for lab in (2, 1, 0)}\n"
        "examples"
    ),
    md("### Eligible (label 2)"),
    code("show_trial(examples[2])"),
    md("### Excluded (label 1)"),
    code("show_trial(examples[1])"),
    md("### Not relevant (label 0)"),
    code("show_trial(examples[0], chars=600)"),
    md("## 3. How long are the criteria? (drives prompt length and GPU cost)"),
    code(
        "words = lambda s: s.fillna('').str.split().str.len()\n"
        "lengths = pd.DataFrame({\n"
        "    'patient note': words(patients.text).describe(),\n"
        "    'inclusion': words(trials.inclusion_text).describe(),\n"
        "    'exclusion': words(trials.exclusion_text).describe(),\n"
        "    'full criteria': words(trials.criteria_text).describe(),\n"
        "}).round(0)\n"
        "lengths"
    ),
    code(
        "full = words(trials.criteria_text)\n"
        "ax = full.clip(upper=3000).plot.hist(bins=60, figsize=(7, 3.5),\n"
        "    title='Words in eligibility criteria (clipped at 3,000)')\n"
        "ax.set_xlabel('words'); plt.tight_layout()\n"
        "print('share of trials over 1,000 words:', round((full > 1000).mean(), 3))"
    ),
    md(
        "## 4. Inclusion / exclusion split quality\n\n"
        "`both` = both sections found; `unsectioned` = free text with no recognisable headers "
        "(kept whole in `criteria_text`)."
    ),
    code(
        "split = trials.split_status.value_counts().to_frame('trials')\n"
        "split['share'] = (split.trials / len(trials)).round(4)\n"
        "split"
    ),
    code(
        "unsectioned = trials[trials.split_status == 'unsectioned'].nct_id\n"
        "for nct in unsectioned.sample(3, random_state=0):\n"
        "    print(nct, '|', T.loc[nct, 'brief_title'])\n"
        "    print(T.loc[nct, 'criteria_text'][:500]); print('=' * 80)"
    ),
    md("## 5. Structured eligibility fields (usable without an LLM)"),
    code(
        "pd.DataFrame({\n"
        "    'has minimum age': trials.minimum_age_years.notna().mean(),\n"
        "    'has maximum age': trials.maximum_age_years.notna().mean(),\n"
        "    'sex restricted': (trials.gender.isin(['Female', 'Male'])).mean(),\n"
        "    'healthy volunteers known': trials.healthy_volunteers.notna().mean(),\n"
        "}, index=['share of judged trials']).T.round(3)"
    ),
    md(
        "## Observations to carry forward\n\n"
        "- Most judged pairs are *not relevant*; *eligible* vs *excluded* is the hard, "
        "clinically meaningful distinction.\n"
        "- Criteria are long: prompt length, not patient-note length, sets GPU cost.\n"
        "- Age and sex are structured fields; they can be checked by code before any LLM call.\n"
        "- Some criteria are scores or nested logic (e.g. NCT02006511: \"at least 8 points\" "
        "from weighted risk factors), which a flat AND/OR list cannot express (Step 3)."
    ),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
    "name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).with_name("01_explore_trec.ipynb")
nbf.write(nb, out)
print("wrote", out)
