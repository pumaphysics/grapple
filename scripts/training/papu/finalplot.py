import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{upgreek}',
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size

##
## Read df
##
df_model = pd.read_csv("rms_model.csv")
df_truth = pd.read_csv("rms_truth.csv")
df_puppi = pd.read_csv("rms_puppi.csv")

plt.clf()
fig, ax = plt.subplots()

ax.tick_params(axis="x",direction="in",labelsize=15)
ax.tick_params(axis="y",direction="in",labelsize=15)
ax.plot(df_truth['x'], df_truth['rms_truth'], label = "Truth", marker='o', color='limegreen')
ax.plot(df_puppi['x'], df_puppi['rms_puppi'], label = "PUPPI", marker='o', color='cornflowerblue')
ax.plot(df_model['x'], df_model['rms_model'], label = "Model", marker='o', color='indianred')
ax.legend(fontsize=18,frameon=False)
ax.set_xlabel(r'Z $p_\mathrm{T}$ (GeV)',fontsize=18)
ax.set_ylabel(r'$RMS(U_\mathrm{perp})$/(<$U_\mathrm{II}$>/<Z $p_\mathrm{T}$>) (GeV)',fontsize=17)

for ext in ('pdf', 'png'):
    plt.savefig('/nobackup/users/bmaier/scan4/paper_plots/rms_plot_perp.' + ext,bbox_inches='tight')
