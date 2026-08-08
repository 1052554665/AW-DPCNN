# replace Elsevier with Springer `sn-article.tex` template
- replace this paper `manuscript.tex` with Springer template `sn-article.tex`

## Conversion Summary: Elsevier `cas-dc` → Springer `sn-jnl`

### 1. Document Class & Preamble
| Before | After |
|---|---|
| `\documentclass[a4paper,fleqn]{cas-dc}` | `\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}` |
| `\usepackage[numbers]{natbib}` | Removed (loaded internally by `sn-jnl`) |
| — | Added Springer-standard packages: `amsthm`, `mathrsfs`, `appendix`, `xcolor`, `manyfoot`, `listings` |
| — | Added Springer theorem styles (`thmstyleone/two/three`) |

### 2. Title & Author Block
- **Title**: `\title [mode = title]{...}` → `\title[...]{...}` (Springer format)
- **Authors**: `\author[affil]{Name}` → `\author[affil]{\fnm{First} \sur{Last}}` (split given/family names)
- **Corresponding author**: `\cormark[1]` + `\ead{...}` → `\author*[...]{...}\email{...}`
- **Affiliations**: `\affiliation[...]{organization=..., city=..., ...}` → `\affil[...]{\orgdiv{...}, \orgname{...}, \orgaddress{...}}`

### 3. Removed (Elsevier-specific)
| Removed | Disposition |
|---|---|
| `\shorttitle{...}`, `\shortauthors{...}` | Not needed in Springer |
| `\tnotemark[1]`, `\tnotetext[1]{...}` | Funding moved to `\bmhead{Acknowledgements}` |
| `\cormark[1]`, `\cortext[cor1]{...}` | Replaced by `\author*` + `\email` |
| `\begin{graphicalabstract}...\end{graphicalabstract}` | Removed (framework figure already in body) |
| `\begin{highlights}...\end{highlights}` | Removed |
| `\let\WriteBookmarks\relax`, `\def\floatpagepagefraction{1}`, `\def\textpagefraction{.001}` | Not needed |
| `\printcredits` | Not needed |
| `\bibliographystyle{unsrtnat}` | Style set by class option |
| `\bio{}...\endbio` | Removed |

### 4. Abstract & Keywords
- `\begin{abstract}...\end{abstract}` → `\abstract{...}`
- `\begin{keywords}...\sep...\end{keywords}` → `\keywords{..., ...}`

### 5. End Matter
- Added `\backmatter` with `\bmhead{Acknowledgements}` (containing funding info)
- Added `\section*{Declarations}` with all required statements
- Bibliography: `\bibliography{references}` retained; sn-mathphys-num.bst copied to paper

### 6. Files Added
- sn-mathphys-num.bst — copied from `sn-article-template/bst/`


> transform the two tables to `sidewaystable` format and put them in one page (keep two tables).