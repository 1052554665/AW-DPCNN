# replace IEEEtran with Elsevier `cas-dc` template
- replace this paper `manuscript.tex` with elsevier template `cas-dc-sample.tex`
- follow the instructions in `elsevier-format-check.md`
- Save files in an editable format, using `.tex` for LaTeX files, consequently, relative data are required to rewrite to the paper rather than cite the `.tex` file.


## Summary of Changes: IEEEtran → Elsevier `cas-dc`

### Files Modified/Created

| File | Action |
|---|---|
| manuscript.tex | **Replaced** — now uses Elsevier `cas-dc` format |
| manuscript-ieee-backup.tex | **Created** — backup of original IEEE version |
| cas-dc.cls | **Copied** from `els-cas-templates/` |
| cas-model2-names.bst | **Copied** from `els-cas-templates/` |

### Key Structural Changes

1. **Document class**: `IEEEtran` → `cas-dc` (Elsevier double-column)
2. **Bibliography**: `IEEEtran.bst` → `cas-model2-names.bst` with `[numbers]{natbib}`
3. **Front matter** now uses Elsevier-style commands:
   - `\title[mode=title]{...}` with `\tnotemark`/`\tnotetext` for funding
   - Individual `\author[]{}` / `\affiliation[]{}` / `\cormark[]` / `\ead{}` per author
   - `\shorttitle{}` / `\shortauthors{}` for running headers

4. **Added required Elsevier elements**:
   - `\begin{highlights}...\end{highlights}` (5 bullet points, each ≤85 chars)
   - `\begin{keywords}...\end{keywords}` with `\sep` separators
   - `\begin{graphicalabstract}...\end{graphicalabstract}` (commented out — needs image)
   - `\printcredits` for author contributions
   - `\bio{}...\endbio` for author biographies (placeholder)

5. **All `\input{}` calls replaced** with inlined table data (6 tables total):
   - Representation comparison table
   - Backbone comparison table
   - Ablation study table
   - Noise robustness table (appendix)
   - Generalization summary table (appendix)
   - Per-class ROC-AUC table (appendix)

### ⚠️ Action Items for You

1. **Fill in affiliations** (lines 53–73): Replace `Your University`, `Your City`, etc. with actual institution details
2. **Fill in corresponding author email** (line 47): Replace `chenyang@example.edu.cn`
3. **Fill in author biographies** (end of file): Replace placeholder comments with actual bios
4. **Graphical abstract** (line 92–94): Uncomment and provide image (531×1328 px recommended)
5. **Compile** with `pdflatex` + `bibtex` to verify everything works with the `cas-dc` class

Made changes.


>compile this file and fix the issue

### Issues Found & Fixed

| # | Error | Fix |
|---|---|---|
| 1 | **Missing thumbnails**: `File 'thumbnails/cas-email.jpeg' not found` — the `cas-dc` class requires thumbnail icons for email/URL links in the author block | Copied `els-cas-templates/thumbnails/` → thumbnails |
| 2 | **Font expansion fatal error**: `auto expansion is only possible with scalable fonts` — `microtype`'s font expansion feature conflicts with `cas-dc`'s font setup | Changed `\usepackage{microtype}` → `\usepackage[expansion=false]{microtype}` |

### Build Result
- **PDF generated**: `manuscript.pdf` (11 MB)
- **0 undefined references**
- **0 undefined citations** (all BibTeX entries resolved)
- Full cycle: `pdflatex` → `bibtex` → `pdflatex` × 2 completed cleanly

Made changes.

>Here are the two ways to compile it:


### Option 1: One-command build (recommended)

```bash
cd /home/yangchen/git_clone/AW-DPCNN/paper
latexmk -pdf manuscript.tex
```

### Option 2: Manual build (if latexmk has issues)

```bash
cd /home/yangchen/git_clone/AW-DPCNN/paper
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

The earlier `latexmk` timeout (exit code 148) was caused by the missing `thumbnails/` folder and the microtype font expansion conflict — both are now fixed, so `latexmk -pdf manuscript.tex` should work cleanly now.


# title page

>move the following information in `manuscript.tex` to `TitlePage.tex`, and review the `TitlePage.tex` file using `title-page.md` to ensure it meets the requirements of the title page information.

```latex
\author[1]{Chen Yang}
\author[2]{Zonglong Bai}
\cormark[1]
\ead{zlbai@ncepu.edu.cn}
\author[3]{Zhiyuan Xie}
\author[4]{Chenggang Liu}
\author[5]{Junyan Zhang}
\author[6]{Yihe Guo}

\affiliation[1]{organization={Department of Electronic and Communication Engineering, North China Electric Power University},
    city={Baoding},
    postcode={071003},
    country={China}}

\affiliation[2]{organization={Department of Electronic and Communication Engineering, North China Electric Power University},
    city={Baoding},
    postcode={071003},
    country={China}}

\affiliation[3]{organization={Department of Electronic and Communication Engineering, North China Electric Power University},
    city={Baoding},
    postcode={071003},
    country={China}}

\affiliation[4]{organization={Department of Electronic and Communication Engineering, North China Electric Power University},
    city={Baoding},
    postcode={071003},
    country={China}}

\affiliation[5]{organization={Department of Electronic and Communication Engineering, North China Electric Power University},
    city={Baoding},
    postcode={071003},
    country={China}}

\affiliation[6]{organization={Department of Electronic and Communication Engineering, North China Electric Power University},
    city={Baoding},
    postcode={071003},
    country={China}}

\cortext[cor1]{Corresponding author.}
```

### Changes to TitlePage.tex

| Element | Before | After |
|---|---|---|
| **Title** | "Voiceprint Diagnosis Method for Transformer Faults..." | "Adaptive Multi-Representation Fusion via Dual-Channel PCNN..." (matches manuscript.tex) |
| **Authors** | 4 authors (missing Zhiyuan Xie) | 6 authors: Chen Yang, Zonglong Bai, **Zhiyuan Xie** (added), Chenggang Liu, Junyan Zhang, Yihe Guo |
| **Affiliation markers** | Numbers (¹²³⁴) | Lower-case letters (ᵃ) per `title-page.md` |
| **Author order** | Non-standard | Matches manuscript.tex exactly |

### Compliance with `title-page.md`

| Requirement | Status |
|---|---|
| Concise, informative title with no abbreviations | ✅ |
| Given + family name, order matches submission | ✅ |
| Lower-case superscript letter for affiliations | ✅ Fixed (all `ᵃ` — same institution) |
| Full postal address with country | ✅ |
| Email for each author (where available) | ✅ |
| Corresponding author clearly indicated | ✅ Zonglong Bai with `zlbai@ncepu.edu.cn` |
| Present/permanent address footnote | N/A — no author has moved |

### ⚠️ Note
**Zhiyuan Xie** (author 3) has no email or ORCID in TitlePage.tex. If you have that information, add it after line `\textbf{Zhiyuan Xie}\textsuperscript{a}\\`.

Made changes.


>revise the cover letter


>revise the highlights according to the instructions in `highlights.md` to fit the Elsevier submission requirements


>this paper will be submitted to digital signal processing, revise relative description to fit the scope of DSP

>revise highlights according to contribution

>revise the cover letter according to the instructions in `coverletter.md` to fit the Elsevier submission requirements