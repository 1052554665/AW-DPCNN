>for this table, delete the `Files​` and `JS Divergence` column, also the signal processing configuration section at the bottom, instead, describe the three parts additionally in the caption. Also, remove the `\multirow` and `\cmidrule` commands, and make sure the table is still readable and clear.


>is it proper to remove the definition of ROC-AUC in IEEE-TIM? 

Short answer: **keep a brief mention, but the full TPR/FPR formulas can go.** IEEE-TIM readers know ROC-AUC. What matters is clarifying your multi-class strategy (OvR + macro-averaging), since that's a design choice with implications for imbalanced data.

Suggested replacement — cut from 9 lines to 3:

```tex
For multi-class ROC-AUC, the One-vs-Rest (OvR) strategy is employed and the macro-averaged AUC (unweighted mean of per-class AUC values) is reported, ensuring balanced assessment across all fault categories regardless of class imbalance.
```

This keeps the *methodological signal* (OvR, macro, justification) while dropping the *textbook formulas* that every reviewer already knows.

>this paper only uses the CWRU dataset, is it proper to focus on non-stationary signals in abstract and introduction? such as rotating machinery, power transformer, rather than only the rotating machinery? dose the range too broad?

**Your current framing is properly calibrated.** Here's why:

| What you claim | What you prove | Verdict |
|---|---|---|
| "Non-stationary signals" | ✅ CWRU signals are non-stationary — this is a *signal property*, not a domain claim | Fine |
| "Rotating machinery" | ✅ Bearings are rotating machinery components | Honest |
| "Power transformers" | ❌ Zero transformer data | Remove any such reference |

The key principle: **claim the problem characteristic, not the application domain.**

Your method (STFT+GADF → AW-DPCNN fusion → MSCA-VGG16) is a *signal-processing pipeline* that works on any non-stationary time series. You're validating it on bearings. That's perfectly legitimate — you're not claiming "this is a bearing-specific method," you're claiming "this is a fusion method for non-stationary signals, validated on bearings."

**Compare to published papers**: Most CWRU-only papers say "rotating machinery fault diagnosis" in the title, not "bearing fault diagnosis." The CWRU dataset is understood as a *benchmark proxy* for rotating machinery faults broadly.

**One nuance**: Your current abstract says "rotating machinery" which is fine. Don't broaden to "industrial machinery" or "power equipment" — those imply validated scope you don't have. The current paper text I've already cleaned up removes transformer claims, so you're in good shape.