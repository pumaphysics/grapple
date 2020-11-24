#import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sys import argv

#sns.set(style="ticks")
#sns.set_context("poster")


mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{upgreek}',
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]

#plt.rc('text', usetex=True)
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size

nbins = 30
bins = np.linspace(0., 300., num=nbins)

##
## Read df
##
#df_model = pd.read_csv("/nobackup/users/bmaier/scan4/64_final_nopuppi_infer_Zvv_highstat/resolution_150_U_model.csv")
df_model = pd.read_csv("/nobackup/users/bmaier/scan4/64_final_nopuppi_infer_Zvv_highstat/resolution_150_U_%s.csv"%argv[2])

##
## Add bin index
##
df_model['bin'] = np.searchsorted(bins, df_model['x'].values)

##
## Add overflow
##
tmp_bins = []
dfvalues = df_model['x'].values.tolist()
dfbins = df_model['bin'].values.tolist()
for i in range(len(df_model['bin'].values.tolist())):
    if dfvalues[i] > bins[-1]:
        tmp_bins.append(nbins)
    else:
        tmp_bins.append(dfbins[i])
df_model['bin'] = tmp_bins


upar_per_bin = {}
uperp_per_bin = {}
x_per_bin = {}


dfupar = df_model['upar'].values.tolist()
dfuperp = df_model['uper'].values.tolist()
dfx = df_model['x'].values.tolist()

for i in np.sort(df_model.bin.unique()):
    for j in range(len(df_model['bin'].values.tolist())):
        if dfbins[j] == i:
            if i not in upar_per_bin:
                upar_per_bin[i] = []
                upar_per_bin[i].append(dfupar[j])
            else:
                upar_per_bin[i].append(dfupar[j])
            if i not in x_per_bin:
                x_per_bin[i] = []
                x_per_bin[i].append(dfx[j])
            else:
                x_per_bin[i].append(dfx[j])

mean_upar_per_bin = []
mean_x_per_bin = []

for i in np.sort(df_model.bin.unique()):
    mean_upar_per_bin.append((-1)*np.mean(np.array(upar_per_bin[i])))
    mean_x_per_bin.append(np.mean(np.array(x_per_bin[i])))

alpha_per_bin = 1./(np.array(mean_upar_per_bin)/np.array(mean_x_per_bin))

for i in np.sort(df_model.bin.unique()):
    for j in range(len(df_model['bin'].values.tolist())):
        if dfbins[j] == i:
            if i not in uperp_per_bin:
                uperp_per_bin[i] = []
                uperp_per_bin[i].append(alpha_per_bin[i-2]*dfuperp[j])
            else:
                uperp_per_bin[i].append(alpha_per_bin[i-2]*dfuperp[j])

rms_uperp_per_bin = []
for i in np.sort(df_model.bin.unique()):
    rms_uperp_per_bin.append(np.sqrt(np.mean(np.square(np.array(uperp_per_bin[i])))))

final_per_bin = rms_uperp_per_bin/(np.array(mean_upar_per_bin)/np.array(mean_x_per_bin))

df = pd.DataFrame.from_dict({'rms_%s'%argv[2]:final_per_bin,'x':mean_x_per_bin})
df.to_csv("rms_%s.csv"%argv[2],index=False)


