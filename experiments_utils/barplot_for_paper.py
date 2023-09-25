import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
bar_width = 0.6
# label_size = 25

sns.set(font_scale=3)

sim_precission = pd.DataFrame(
    {'class': ["C1F1s", "C1F2s", "SE1s", "SE2s", "SE3s"],
     'FSC': [2/4.5, 5.5/5.5, 6.5/7, 3.5/3.5, 10/12.5],
     'Ours': [2/2.5, 5/5, 5.5/5.5, 7/7, 15/15]}
)

sim_recall = pd.DataFrame(
    {'class': ["C1F1s", "C1F2s", "SE1s", "SE2s", "SE3s"],
     'FSC': [2/5, 5.5/8, 6.5/10, 3.5/7, 10/25],
     'Ours': [2/5, 5/8, 5.5/10, 7/7, 15/25]}
)

real_precission = pd.DataFrame(
    {'class': ["C1F1r", "C1F2r", "C2F2r", "C3F2r"],
     'FSC': [3.5/3.5, 5.5/5.5, 11/12.5, 13.5/14.5],
     'Neural oR': [2/3.5, 5.5/5.5, 11.5/11.5, 10/13.5]}
)

real_recall = pd.DataFrame(
    {'class': ["C1F1r", "C1F2r", "C2F2r", "C3F2r"],
     'FSC': [3.5/6.5, 5.5/8, 11/21, 13.5/16],
     'Neural oR': [2/6.5, 5.5/8, 11.5/21, 10/16]}
)

plt.figure("sim Precision")
plt.clf()
sim_precission_groups = sim_precission.melt(id_vars=['class'], var_name=['Metric'])
sim_precission_plot = sns.barplot(data=sim_precission_groups, x='class', y='value', hue='Metric', errorbar=None, palette="viridis", width=bar_width)
# sim_precission_plot.tick_params(labelsize=label_size)
plt.legend([], [], frameon=False)
for item in sim_precission_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("")
plt.ylabel("")
plt.savefig('/home/adminpc/Documents/My papers/ICRA 2024/Images/sim precision.png', bbox_inches='tight')

plt.figure("sim Recall")
plt.clf()
sim_recall_groups = sim_recall.melt(id_vars=['class'], var_name=['Metric'])
sim_recall_plot = sns.barplot(data=sim_recall_groups, x='class', y='value', hue='Metric', errorbar=None, palette="viridis", width=bar_width)
# sim_recall_plot.tick_params(labelsize=label_size)
plt.legend([], [], frameon=False)
for item in sim_recall_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("")
plt.ylabel("")
plt.savefig('/home/adminpc/Documents/My papers/ICRA 2024/Images/sim recall.png', bbox_inches='tight')

plt.figure("real Precision")
plt.clf()
real_precission_groups = real_precission.melt(id_vars=['class'], var_name=['Metric'])
real_precission_plot = sns.barplot(data=real_precission_groups, x='class', y='value', hue='Metric', errorbar=None, palette="viridis", width=bar_width)
# real_precission_plot.tick_params(labelsize=label_size)
plt.legend([], [], frameon=False)
for item in real_precission_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("")
plt.ylabel("")
plt.savefig('/home/adminpc/Documents/My papers/ICRA 2024/Images/real precision.png', bbox_inches='tight')

plt.figure("real Recall")
plt.clf()
real_recall_groups = real_recall.melt(id_vars=['class'], var_name=['Metric'])
real_recall_plot = sns.barplot(data=real_recall_groups, x='class', y='value', hue='Metric', errorbar=None, palette="viridis", width=bar_width)
# real_recall_plot.tick_params(labelsize=label_size)
plt.legend([], [], frameon=False)
for item in real_recall_plot.get_xticklabels():
    item.set_rotation(45)
plt.xlabel("")
plt.ylabel("")
plt.savefig('/home/adminpc/Documents/My papers/ICRA 2024/Images/real recall.png', bbox_inches='tight')


plt.show()
