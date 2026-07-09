>review and revise all the tables in the paper to ensure they follow IEEE style guidelines. Make sure that the captions are placed before the tables, use title case, and that the font size is set to footnotesize. Additionally, verify that the labels are correctly positioned after the captions for proper referencing. The descriptions of the tables can be moved to the main text if they are too long, but ensure that they are still clear and concise. The following official IEEE style guidelines should be followed for all tables in the paper:

```
\section{Tables}
Note that, for IEEE-style tables, the
 $\backslash${\tt{caption}} command should come BEFORE the table. Table captions use title case. Articles (a, an, the), coordinating conjunctions (and, but, for, or, nor), and most short prepositions are lowercase unless they are the first or last word. Table text will default to $\backslash${\tt{footnotesize}} as
 the IEEE normally uses this smaller font for tables.
 The $\backslash${\tt{label}} must come after $\backslash${\tt{caption}} as always.
 
\begin{table}[!t]
\caption{An Example of a Table\label{tab:table1}}
\centering
\begin{tabular}{|c||c|}
\hline
One & Two\\
\hline
Three & Four\\
\hline
\end{tabular}
\end{table}
```

## Summary of Changes

### Files Modified
1. **manuscript.tex** — 3 inline tables
2. **rep_compare_table.tex**
3. **backbone_comparison_table.tex**
4. **ablation_table.tex**

---

### Fixes Applied (per IEEE guidelines)

| Fix | Before | After |
|---|---|---|
| **`\caption` before `\centering`** | `\caption` placed after `\centering` in 4 of 6 tables | `\caption` moved before `\centering` in ALL tables |
| **`\label` embedded in `\caption`** | `\label` on separate line (sometimes after `\end{tabular}`) | `\label{...}` embedded inside `\caption{...\label{...}}` for all tables |
| **Font size** | `\small` used in 4 tables (overriding IEEE default `\footnotesize`) | `\small` removed from all tables — now uses IEEE default `\footnotesize` |
| **Title case** | Inconsistent capitalization, some all-lowercase | Applied IEEE title case rules (short prepositions, articles, coordinating conjunctions lowercase unless first/last) |
| **Overlong captions** | `rep_compare_table` had 3-sentence caption; `ablation_table` had abbreviation glossary in caption | Shortened to concise single-sentence captions (details already in main text) |

---

### Per-Table Details

| Table | Key Changes |
|---|---|
| **`tab:awdpcnn_params`** | `\caption`→before `\centering`; `\label`→embedded; "Where" capitalized |
| **`tab:operating_conditions`** | `\label` moved from after `\end{tabular}` into `\caption`; added "the" before "CWRU" |
| **`tab:datasets_comprehensive`** | Removed `\small`; applied title case ("Class-Wise Sample Distribution of Three CWRU Bearing Datasets"); removed stale commented-out long caption |
| **`tab:rep_compare`** | `\caption`→before `\centering`; `\small` removed; caption shortened from 3 sentences to 1 ("Representation Comparison across Time–Frequency and Temporal Encoding Methods"); "across" lowercase per preposition rule |
| **`tab:network_comparison`** | `\caption`→before `\centering`; `\small` removed |
| **`tab:ablation_unified`** | `\caption`→before `\centering`; `\small` removed; abbreviation glossary (MS/CA/EH) removed from caption (already defined in column headers and main text) |

All `\label` references used in `\ref{}` commands throughout the manuscript remain unchanged and will resolve correctly.

Made changes.

>it seems that all the tables still not apply "title case" for the captions. Please make sure that all table captions follow the IEEE title case rules, where articles (a, an, the), coordinating conjunctions (and, but, for, or, nor), and most short prepositions are lowercase unless they are the first or last word. The following official IEEE style guidelines should be followed for all tables in the paper:

```
for IEEE-style tables, the
 $\backslash${\tt{caption}} command should come BEFORE the table. Table captions use title case. Articles (a, an, the), coordinating conjunctions (and, but, for, or, nor), and most short prepositions are lowercase unless they are the first or last word. Table text will default to $\backslash${\tt{footnotesize}} as
 the IEEE normally uses this smaller font for tables.
 The $\backslash${\tt{label}} must come after $\backslash${\tt{caption}} as always.
```

>review all the tables, it seems that the title and captions of the tables are like `TABLE I AN EXAMPLE OF A TABLE` in published IEEE papers, rather than  `TABLE I: An Example of a Table` in this paper.