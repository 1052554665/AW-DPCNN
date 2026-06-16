>convert `mst.tex` from iopjournal to `IEEEtran` format, keeping all content intact.


### Changes Made

| Element | Before (`iopjournal`) | After (`IEEEtran`) |
|---------|----------------------|---------------------|
| Document class | `\documentclass[anonymous]{iopjournal}` | `\documentclass[journal]{IEEEtran}` |
| Package | — | Added `\usepackage{balance}` (required by IEEEtran) |
| Hyphenation | — | Added `\hyphenation{...}` (IEEE convention) |
| Author block | None (anonymous) | `\author{...}` with `\IEEEmembership{}`, `\thanks{}` placeholders |
| Running heads | None | `\markboth{Journal name}{Authors: Title}` |
| `\maketitle` | Implicit | Explicit `\maketitle` after author block |
| Keywords | `\keywords{...}` | `\begin{IEEEkeywords}...\end{IEEEkeywords}` |
| First paragraph | Plain text | `\IEEEPARstart{P}{ower}...` (IEEE drop-cap) |
| Bibliography | `\bibliographystyle{iopart-num}` | Removed (IEEEtran uses its own style) |

### ⚠️ You MUST fill in these placeholders before submission:

1. **Author block** (line ~24–30): Replace `Author~One`, `Author~Two`, `Author~Three` with real names. Update `\IEEEmembership{}` if any author is an IEEE member. Fill in `\thanks{}` with manuscript dates, funding info, and affiliations.

2. **`\markboth`** (line ~32): Change `IEEE Transactions on Instrumentation and Measurement` to your target journal name.

3. **Bibliography**: If you're using BibTeX, `references.bib` should already work. If using manual `\bibitem` entries, ensure they follow IEEE format (see the template for examples).

### Compilation

Compile with the standard IEEEtran workflow:
```bash
pdflatex mst
bibtex mst
pdflatex mst
pdflatex mst
```

The `IEEEtran.cls` is bundled with most TeX distributions (TeX Live, MiKTeX). The existing `iopjournal.cls` and `iopart-num.bst` files in the Paper directory can be safely removed or kept for reference — they won't affect IEEEtran compilation.

Made changes.