# multi subplot of model comparison

compare model performance using `test_auc` from the experiment results. refer to `test_log.csv` in `experiment_result` of each model, plot `test_auc`. Just like the following example:


### Step 1: Prepare your CSV data structure
```
machine_type,team_rank,auc,pauc
Toy-car,1,0.95,0.90
Toy-car,2,0.93,0.88
Toy-conveyor,1,0.85,0.75
...
```

### Step 2: Create the plot in Python

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV
df = pd.read_csv('your_data.csv')

# Setup figure with subplots (top=AUC, bottom=pAUC)
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Define machine types and colors
machines = ['Toy-car', 'Toy-conveyor', 'Fan', 'Pump', 'Slide rail', 'Valve']
colors = ['#4A90E2', '#7B68EE', '#50E3C2', '#F5A623', '#BD10E0', '#50E3C2']

# Top plot: Average AUC
for idx, machine in enumerate(machines):
    machine_data = df[df['machine_type'] == machine]
    axes[0].bar(machine_data['team_rank'] + idx*0.12, 
               machine_data['auc'], 
               width=0.12, 
               label=machine,
               color=colors[idx])

axes[0].axhline(y=0.85, color='white', linestyle='--', linewidth=2, label='Baseline')
axes[0].set_ylabel('Average AUC [%]', fontsize=12)
axes[0].set_ylim([50, 100])
axes[0].legend(loc='upper right', ncol=6)
axes[0].set_title('Evaluation results of top 10 teams in team ranking', fontsize=14)
axes[0].grid(axis='y', alpha=0.3)

# Bottom plot: Average pAUC
for idx, machine in enumerate(machines):
    machine_data = df[df['machine_type'] == machine]
    axes[1].bar(machine_data['team_rank'] + idx*0.12, 
               machine_data['pauc'], 
               width=0.12, 
               label=machine,
               color=colors[idx])

axes[1].axhline(y=0.65, color='white', linestyle='--', linewidth=2, label='Baseline')
axes[1].set_xlabel('Total Team Rank', fontsize=12)
axes[1].set_ylabel('Average pAUC [%]', fontsize=12)
axes[1].set_ylim([50, 100])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
plt.show()
```


# Grouped bar chart of model comparison
>for each model, plot params, auc, acc, G-mean, the same metrics using the same style.

The grouped bar chart is ready at model_comparison_grouped.png:

- **x-axis**: Models (MSCA-VGG16 first, then sorted by accuracy)
- **Grouped bars per model**: Acc (dark solid), F1 (medium `//`), AUC (light `\\`), G-Mean (very light `xx`)
- **Below x-axis**: Parameter count (e.g. "26.8M")
- **Reference line**: Chance level (10%)
- **Annotations**: Values above each bar

All grayscale + hatch patterns — IEEE publication-ready.

Made changes.

