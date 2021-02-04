import torch
from torch import nn 
from .utils import t2n 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
import pandas as pd
from loguru import logger
import random

import pyjet

EPS = 1e-4

def kl_divergence(p, q, epsilon=1e-8):
    '''
    epsilon to avoid log(0) or divided by 0 error
    '''
    d_kl = (p * torch.log((p / (q+ epsilon)) + epsilon)).sum()
    return d_kl

def collapsehist(histu,histgm):
    containszero = False
    for e in histu:
        if e == 0:
            containszero = True
            break

    while containszero:
        tmp_histu = []
        tmp_histgm = []
        endreached = False
        for e in range(len(histu)):
            if endreached:
                break
            if e == len(histu)-1 and histu[e] != 0:
                tmp_histu.append(histu[e])
                tmp_histgm.append(histgm[e])
                break
            elif e == len(histu)-1:
                tmp_histgm[-1] += histgm[e]
                break
            if histu[e] == 0:
                for f in range(e+1,len(histu)):
                    if f == e+1:
                        tmp_histu.append(histu[e]+histu[f])
                        tmp_histgm.append(histgm[e]+histgm[f])
                    else:
                        tmp_histu.append(histu[f])
                        tmp_histgm.append(histgm[f])
                    if f == len(histu)-1:
                        endreached = True
                        break
            else:
                tmp_histu.append(histu[e])
                tmp_histgm.append(histgm[e])

        histu = tmp_histu
        histgm = tmp_histgm

        containszero = False
        for e in histu:
            if e == 0:
                containszero = True
                break

    return histu,histgm

class Metrics(object):
    def __init__(self, device, softmax=True):
        self.loss_calc = nn.CrossEntropyLoss(
                ignore_index=-1, 
                reduction='none'
                # weight=torch.FloatTensor([1, 5]).to(device)
            )
        self.reset()
        self.apply_softmax = softmax

    def reset(self):
        self.loss = 0 
        self.acc = 0 
        self.pos_acc = 0
        self.neg_acc = 0
        self.n_pos = 0
        self.n_particles = 0
        self.n_steps = 0
        self.hists = {}

    @staticmethod
    def make_roc(pos_hist, neg_hist):
        pos_hist = pos_hist / pos_hist.sum()
        neg_hist = neg_hist / neg_hist.sum()
        tp, fp = [], []
        for i in np.arange(pos_hist.shape[0], -1, -1):
            tp.append(pos_hist[i:].sum())
            fp.append(neg_hist[i:].sum())
        auc = np.trapz(tp, x=fp)
        plt.plot(fp, tp, label=f'AUC={auc:.3f}')
        return fp, tp

    def add_values(self, yhat, y, label, idx, w=None):
        if w is not None:
            w = w[y==label]
        hist, self.bins = np.histogram(yhat[y==label], bins=np.linspace(0, 1, 100), weights=w)
        if idx not in self.hists:
            self.hists[idx] = hist + EPS
        else:
            self.hists[idx] += hist

    def compute(self, yhat, y, orig_y, w=None, m=None):
        # yhat = [batch, particles, labels]; y = [batch, particles]
        loss = self.loss_calc(yhat.view(-1, yhat.shape[-1]), y.view(-1))
        if w is not None:
            wv = w.view(-1)
            loss *= wv
        if m is None:
            m = np.ones_like(t2n(y), dtype=bool)
        loss = torch.mean(loss)
        self.loss += t2n(loss).mean()

        mask = (y != -1)
        n_particles = t2n(mask.sum())

        pred = torch.argmax(yhat, dim=-1) # [batch, particles]
        pred = t2n(pred)
        y = t2n(y)
        mask = t2n(mask)

        acc = (pred == y)[mask].sum() / n_particles 
        self.acc += acc

        n_pos = np.logical_and(m, y == 1).sum()
        pos_acc = (pred == y)[np.logical_and(m, y == 1)].sum() / n_pos
        self.pos_acc += pos_acc
        neg_acc = (pred == y)[np.logical_and(m, y == 0)].sum() / (n_particles - n_pos)
        self.neg_acc += neg_acc

        self.n_pos += n_pos
        self.n_particles += n_particles

        self.n_steps += 1

        if self.apply_softmax:
            yhat = t2n(nn.functional.softmax(yhat, dim=-1))
        else:
            yhat = t2n(yhat)
        if w is not None:
            w = t2n(w).reshape(orig_y.shape)
            wm = w[m]
            wnm = w[~m]
        else:
            wm = wnm = None
        self.add_values(yhat[:,:,1][m], orig_y[m], 0, 0, wm)
        self.add_values(yhat[:,:,1][m], orig_y[m], 1, 1, wm)
        self.add_values(yhat[:,:,1][~m], orig_y[~m], 0, 2, wnm)
        self.add_values(yhat[:,:,1][~m], orig_y[~m], 1, 3, wnm)

        #if self.n_steps % 50 == 0 and False:
        #    print(t2n(y[0])[:10])
        #    print(t2n(pred[0])[:10])
        #    print(t2n(yhat[0])[:10, :])

        return loss, acc

    def mean(self):
        return ([x / self.n_steps 
                 for x in [self.loss, self.acc, self.pos_acc, self.neg_acc]]
                + [self.n_pos / self.n_particles])

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,
                'bins': self.bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists[0], label='PU Neutral', **hist_args)
        plt.hist(weights=self.hists[1], label='Hard Neutral', **hist_args)
        plt.hist(weights=self.hists[2], label='PU Charged', **hist_args)
        plt.hist(weights=self.hists[3], label='Hard Charged', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel('P(Hard|p,e)')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        plt.clf()
        fig_handle = plt.figure()
        fp, tp, = self.make_roc(self.hists[1], self.hists[0])
        plt.ylabel('True Neutral Positive Rate')
        plt.xlabel('False Neutral Positive Rate')
        path += '_roc'
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)
        pickle.dump({'fp': fp, 'tp': tp}, open(path + '.pkl', 'wb'))
        plt.close(fig_handle)


class METMetrics(Metrics):
    def __init__(self, device, softmax=True):
        super().__init__(device, softmax)

        self.mse = nn.MSELoss()
        self.met_loss_weight = 1

    def compute(self, yhat, y, orig_y, met, methat, w=None, m=None):
        met_loss = self.met_loss_weight * self.mse(methat.view(-1), met.view(-1))
        pu_loss, acc = super().compute(yhat, y, orig_y, w, m)
        loss = met_loss + pu_loss 
        return loss, acc


class JetResolution(object):
    def __init__(self):
        self.bins = {
                'pt': np.linspace(-100, 100, 100),
                'm': np.linspace(-50, 50, 100),
                'mjj': np.linspace(-1000, 1000, 100),
            }
        self.bins_2 = {
                'pt': np.linspace(0, 500, 100),
                'm': np.linspace(0, 50, 100),
                'mjj': np.linspace(0, 3000, 100),
            }
        self.labels = {
                'pt': 'Jet $p_T$',
                'm': 'Jet $m$',
                'mjj': '$m_{jj}$',
            }
        self.reset()

    def reset(self):
        self.dists = {k:[] for k in ['pt', 'm', 'mjj']} 
        self.dists_2 = {k:([], []) for k in ['pt', 'm', 'mjj']} 

    @staticmethod 
    def compute_mass(x):
        pt, eta, e = x[:, :, 0], x[:, :, 1], x[:, :, 3]
        p = pt * np.cosh(eta)
        m = np.sqrt(np.clip(e**2 - p**2, 0, None))
        return m

    def compute(self, x, weight, mask, pt0, m0, mjj):
        x = np.copy(x[:, :, :4])
        m = self.compute_mass(x)
        x[:, :, 0] = x[:, :, 0] * weight
        x[:,:,3] = m
        #print(x[:,:,3])
        #x[:, :, 3] = 0 # temporary override to approximate mass 
        x = x.astype(np.float64)
        n_batch = x.shape[0] 
        for i in range(n_batch):
            evt = x[i][np.logical_and(mask[i].astype(bool), x[i,:,0]>0)]
            evt = np.core.records.fromarrays(
                    evt.T, 
                    names='pt, eta, phi, m',
                    formats='f8, f8, f8, f8'
                )
            seq = pyjet.cluster(evt, R=0.4, p=-1)
            jets = seq.inclusive_jets()
            if len(jets) > 0:
                self.dists['pt'].append(jets[0].pt - pt0[i])
                self.dists_2['pt'][0].append(pt0[i])
                self.dists_2['pt'][1].append(jets[0].pt)

                self.dists['m'].append(jets[0].mass - m0[i])
                self.dists_2['m'][0].append(m0[i])
                self.dists_2['m'][1].append(jets[0].mass)

                if len(jets) > 1:
                    j0, j1 = jets[:2]
                    mjj_pred = np.sqrt(
                            (j0.e + j1.e) ** 2 
                            - (j0.px + j1.px) ** 2
                            - (j0.py + j1.py) ** 2
                            - (j0.pz + j1.pz) ** 2
                        )
                else:
                    mjj_pred = 0 
                if mjj[i] > 0:
                    self.dists['mjj'].append(mjj_pred - mjj[i])
                    self.dists_2['mjj'][0].append(mjj[i])
                    self.dists_2['mjj'][1].append(mjj_pred)

    def plot(self, path):
        for k, data in self.dists.items():
            plt.clf()
            plt.hist(data, bins=self.bins[k])
            plt.xlabel(f'Predicted-True {self.labels[k]} [GeV]')
            for ext in ('pdf', 'png'):
                plt.savefig(f'{path}_{k}_err.{ext}')
            with open(f'{path}_{k}_err.pkl', 'wb') as fpkl:
                pickle.dump(
                        {'data': data, 'bins':self.bins[k]},
                        fpkl
                    )

        for k, data in self.dists_2.items():
            plt.clf()
            plt.hist2d(data[0], data[1], bins=self.bins_2[k])
            plt.xlabel(f'True {self.labels[k]} [GeV]')
            plt.ylabel(f'Predicted {self.labels[k]} [GeV]')
            for ext in ('pdf', 'png'):
                plt.savefig(f'{path}_{k}_corr.{ext}')



class METResolution(object):
    def __init__(self, bins=np.linspace(-100, 100, 40)):
        self.bins = bins
        self.bins_2 = (0, 300)
        self.bins_met1 = (0, 300)
        self.df = None
        self.df_p = None
        self.df_model = None
        self.df_truth = None
        self.df_puppi = None
        self.reset()

    def reset(self):
        self.dist = None
        self.dist_p = None
        self.dist_pup = None
        self.dist_2 = None
        self.dist_met = None
        self.dist_pred = None
        self.dist_2_p = None
        self.dist_2_pup = None

    @staticmethod
    def _compute_res(pt, phi, w, gm):
        pt = pt * w
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        metx = np.sum(px, axis=-1)
        mety = np.sum(py, axis=-1)
        met = np.sqrt(np.power(metx, 2) + np.power(mety, 2))
        res = (met / gm) - 1
        return res

    def compute(self, pf, pup, gm, pred, weight=None):
        res = (pred - gm)
        res_p = (pf - gm) 
        res_pup = (pup - gm)

        hist, _ = np.histogram(res, bins=self.bins)
        hist_met, self.bins_met = np.histogram(gm, bins=np.linspace(*(self.bins_met1) + (100,)))
        hist_pred, self.bins_pred = np.histogram(pred, bins=np.linspace(*(self.bins_met1) + (100,)))
        hist_p, _ = np.histogram(res_p, bins=self.bins)
        hist_pup, _ = np.histogram(res_pup, bins=self.bins)
        hist_2, _, _ = np.histogram2d(gm, res, bins=100, range=(self.bins_2, self.bins_2))
        hist_2_p, _, _ = np.histogram2d(gm, p, bins=100, range=(self.bins_2, self.bins_2))
        hist_2_pup, _, _ = np.histogram2d(gm, pup, bins=100, range=(self.bins_2, self.bins_2))
        if self.dist is None:
            self.dist = hist + EPS
            self.dist_met = hist_met + EPS
            self.dist_pred = hist_pred + EPS
            self.dist_p = hist_p + EPS
            self.dist_pup = hist_pup + EPS
            self.dist_2 = hist_2 + EPS
            self.dist_2_p = hist_2_p + EPS
            self.dist_2_pup = hist_2_pup + EPS
        else:
            self.dist += hist
            self.dist_met += hist_met
            self.dist_pred += hist_pred
            self.dist_p += hist_p
            self.dist_pup += hist_pup
            self.dist_2 += hist_2
            self.dist_2_p += hist_2_p
            self.dist_2_pup += hist_2_pup

    @staticmethod 
    def _compute_moments(x, dist):
        dist = dist / np.sum(dist)
        mean = np.sum(x * dist) 
        var = np.sum(np.power(x - mean, 2) * dist) 
        return mean, var

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5

        mean, var = self._compute_moments(x, self.dist)
        mean_p, var_p = self._compute_moments(x, self.dist_p)
        mean_pup, var_pup = self._compute_moments(x, self.dist_pup)

        label = r'Model ($\delta=' + f'{mean:.1f}' + r'\pm' + f'{np.sqrt(var):.1f})$'
        plt.hist(x=x, weights=self.dist, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{mean_pup:.1f}' + r'\pm' + f'{np.sqrt(var_pup):.1f})$'
        plt.hist(x=x, weights=self.dist_pup, label=label, histtype='step', bins=self.bins)

        label = r'Ground Truth ($\delta=' + f'{mean_p:.1f}' + r'\pm' + f'{np.sqrt(var_p):.1f})$'
        plt.hist(x=x, weights=self.dist_p, label=label, histtype='step', bins=self.bins, linestyle='--')

        plt.xlabel('Predicted-True MET [GeV]')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        plt.clf()
        x = (self.bins_met[:-1] + self.bins_met[1:]) * 0.5
        plt.hist(x=x, weights=self.dist_met, bins=self.bins_met)
        plt.xlabel('True MET')
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_true.' + ext)

        plt.clf()
        x = (self.bins_pred[:-1] + self.bins_pred[1:]) * 0.5
        plt.hist(x=x, weights=self.dist_pred, bins=self.bins_pred)
        plt.xlabel('Predicted MET')
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_pred.' + ext)

        plt.clf()
        self.dist_2 = np.ma.masked_where(self.dist_2 < 0.5, self.dist_2)
        plt.imshow(self.dist_2.T, vmin=0.5, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.xlabel('True MET [GeV]')
        plt.ylabel('Predicted MET [GeV]')
        plt.colorbar()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_corr.' + ext)
        plt.clf()
        self.dist_2_p = np.ma.masked_where(self.dist_2_p < 0.5, self.dist_2_p)
        plt.imshow(self.dist_2_p.T, vmin=0.5, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.colorbar()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_corr_pf.' + ext)
        self.reset()

        return {'model': (mean, np.sqrt(var)), 'puppi': (mean_p, np.sqrt(var_p))}


class ParticleMETResolution(METResolution):
    @staticmethod
    def _compute_res(pt, phi, w, gm, gmphi):
        pt = pt * w
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        metx = (-1)*np.sum(px, axis=-1)
        mety = (-1)*np.sum(py, axis=-1)
        met = np.sqrt(np.power(metx, 2) + np.power(mety, 2))
        gmx = gm * np.cos(gmphi)
        gmy = gm * np.sin(gmphi)
        res =  met - gm # (met / gm) - 1        
        resx = metx - gmx
        resy = mety - gmy
        return res,resx,resy

    def compute(self, pt, phi, w, y, baseline, gm, gmphi):
        res,resx,resy = self._compute_res(pt, phi, w, gm, gmphi)
        res_t,resx_t,resy_t = self._compute_res(pt, phi, y, gm, gmphi)
        res_p,resx_p,resy_p = self._compute_res(pt, phi, baseline, gm, gmphi)

        #data = {'x': res, 'col_2': ['a', 'b', 'c', 'd']}
        df = pd.DataFrame()
        df_p = pd.DataFrame()
        df['y'] = res
        df_p['y'] = res_p
        df['x'] = gm
        df_p['x'] = gm

        bins = np.linspace(0., 300., num=25)
        df['bin'] = np.searchsorted(bins, df['x'].values)
        df_p['bin'] = np.searchsorted(bins, df_p['x'].values)

        hist, _ = np.histogram(res, bins=self.bins)
        histx, _ = np.histogram(resx, bins=self.bins)
        histy, _ = np.histogram(resy, bins=self.bins)
        hist_p, _ = np.histogram(res_p, bins=self.bins)
        histx_p, _ = np.histogram(resx_p, bins=self.bins)
        histy_p, _ = np.histogram(resy_p, bins=self.bins)
        hist_met, _ = np.histogram(res_t, bins=self.bins)
        histx_met, _ = np.histogram(resx_t, bins=self.bins)
        histy_met, _ = np.histogram(resy_t, bins=self.bins)
        hist_2, _, _ = np.histogram2d(gm, res+gm, bins=25, range=(self.bins_2, self.bins_2))        
        hist_2_p, _, _ = np.histogram2d(gm, res_p+gm, bins=25, range=(self.bins_2, self.bins_2))        
        #hist_2, xedges, _ = np.histogram2d(gm, res, bins=25, range=(self.bins_2, self.bins_2))        
        #hist_2_p, _, _ = np.histogram2d(gm, res_p, bins=25, range=(self.bins_2, self.bins_2))        

        self.xedges = bins

        #print(res)
        #print(self.xedges)

        if self.df is None:
            self.df = df
            self.df_p = df_p
        else:
            tmp = pd.concat([self.df,df],ignore_index=True,sort=False)
            tmp_p = pd.concat([self.df_p,df_p],ignore_index=True,sort=False)
            #tmp_p = self.df_p.append(df_p)
            self.df = tmp
            self.df_p = tmp_p
            
        if self.dist is None:
            self.dist = hist + EPS
            self.distx = histx + EPS
            self.disty = histy + EPS
            self.dist_p = hist_p + EPS
            self.distx_p = histx_p + EPS
            self.disty_p = histy_p + EPS
            self.dist_met = hist_met + EPS
            self.distx_met = histx_met + EPS
            self.disty_met = histy_met + EPS
            self.dist_2 = hist_2 + EPS
            self.dist_2_p = hist_2_p + EPS
        else:
            self.dist += hist
            self.distx += histx
            self.disty += histy
            self.dist_p += hist_p
            self.distx_p += histx_p
            self.disty_p += histy_p
            self.dist_met += hist_met
            self.distx_met += histx_met
            self.disty_met += histy_met
            self.dist_2 += hist_2
            self.dist_2_p += hist_2_p

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5

        mean, var = self._compute_moments(x, self.dist)
        meanx, varx = self._compute_moments(x, self.distx)
        meany, vary = self._compute_moments(x, self.disty)
        mean_p, var_p = self._compute_moments(x, self.dist_p)
        meanx_p, varx_p = self._compute_moments(x, self.distx_p)
        meany_p, vary_p = self._compute_moments(x, self.disty_p)
        mean_met, var_met = self._compute_moments(x, self.dist_met)
        meanx_met, varx_met = self._compute_moments(x, self.distx_met)
        meany_met, vary_met = self._compute_moments(x, self.disty_met)

        label = r'Model ($\delta=' + f'{mean:.1f}' + r'\pm' + f'{np.sqrt(var):.1f})$'
        plt.hist(x=x, weights=self.dist, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{mean_p:.1f}' + r'\pm' + f'{np.sqrt(var_p):.1f})$'
        plt.hist(x=x, weights=self.dist_p, label=label, histtype='step', bins=self.bins)

        label = r'Truth+PF ($\delta=' + f'{mean_met:.1f}' + r'\pm' + f'{np.sqrt(var_met):.1f})$'
        plt.hist(x=x, weights=self.dist_met, label=label, histtype='step', bins=self.bins)

        plt.xlabel('(Predicted - True)')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        plt.clf()
        fig, ax = plt.subplots()

        label = r'Model ($\delta=' + f'{meanx:.1f}' + r'\pm' + f'{np.sqrt(varx):.1f})$'
        plt.hist(x=x, weights=self.distx, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{meanx_p:.1f}' + r'\pm' + f'{np.sqrt(varx_p):.1f})$'
        plt.hist(x=x, weights=self.distx_p, label=label, histtype='step', bins=self.bins)

        label = r'Truth+PF ($\delta=' + f'{meanx_met:.1f}' + r'\pm' + f'{np.sqrt(varx_met):.1f})$'
        plt.hist(x=x, weights=self.distx_met, label=label, histtype='step', bins=self.bins)

        plt.xlabel('(X Predicted - X True)')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_x.' + ext)

        plt.clf()
        fig, ax = plt.subplots()

        label = r'Model ($\delta=' + f'{meany:.1f}' + r'\pm' + f'{np.sqrt(vary):.1f})$'
        plt.hist(x=x, weights=self.distx, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{meany_p:.1f}' + r'\pm' + f'{np.sqrt(vary_p):.1f})$'
        plt.hist(x=x, weights=self.distx_p, label=label, histtype='step', bins=self.bins)

        label = r'Truth+PF ($\delta=' + f'{meany_met:.1f}' + r'\pm' + f'{np.sqrt(vary_met):.1f})$'
        plt.hist(x=x, weights=self.distx_met, label=label, histtype='step', bins=self.bins)

        plt.xlabel('(Y Predicted - Y True)')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_y.' + ext)

        plt.clf()
        fig, ax = plt.subplots()
        plt.imshow(self.dist_2.T, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.xlabel('True $p_T^{miss}$ (GeV)')
        plt.ylabel('PUMA $p_T^{miss}$ (GeV)')
        plt.colorbar()
        fig.tight_layout()

        for ext in ('pdf', 'png'):
            plt.savefig(path + '_2D_puma.' + ext)

        plt.clf()
        fig, ax = plt.subplots()
        plt.imshow(self.dist_2_p.T, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.xlabel('True $p_T^{miss}$ (GeV)')
        plt.ylabel('PUPPI $p_T^{miss}$ (GeV)')
        plt.colorbar()
        fig.tight_layout()

        for ext in ('pdf', 'png'):
            plt.savefig(path + '_2D_puppi.' + ext)



        ''' 
        #print(self.df)
        resp_df_binned = self.df.groupby('bin', as_index=False)['y'].mean()
        print(resp_df_binned)
        print(self.xedges)
        resp_df_p_binned = self.df_p.groupby('bin', as_index=False)['y'].mean()
        std_df_binned = self.df.groupby('bin', as_index=False)['y'].std()
        std_df_p_binned = self.df_p.groupby('bin', as_index=False)['y'].std()

        plt.clf()
        fig, ax = plt.subplots()
        #plt.plot((self.xedges[1:] + self.xedges[:-1]) / 2, resp_df_binned, label = "Model", marker='o')
        #plt.plot((self.xedges[1:] + self.xedges[:-1]) / 2, resp_df_p_binned, label = "PUPPI", marker='o')
        plt.plot(resp_df_binned['bin'], resp_df_binned['y'], label = "Model", marker='o')
        plt.plot(resp_df_p_binned['bin'], resp_df_p_binned['y'], label = "PUPPI", marker='o')
        plt.legend()
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_response.' + ext)

        plt.clf()
        fig, ax = plt.subplots()
        #plt.plot((self.xedges[1:] + self.xedges[:-1]) / 2, std_df_binned, label = "Model")
        #plt.plot((self.xedges[1:] + self.xedges[:-1]) / 2, std_df_p_binned, label = "PUPPI")
        plt.plot(std_df_binned['bin'], std_df_binned['y'], label = "Model", marker='o')
        plt.plot(std_df_p_binned['bin'], std_df_p_binned['y'], label = "PUPPI", marker='o')
        plt.legend()
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_std.' + ext)
        '''

        return {'model': (mean, np.sqrt(var)), 'puppi': (mean_p, np.sqrt(var_p))}


class PapuMetrics(object):
    def __init__(self, beta=False):
        self.beta = beta 
        if not self.beta:
            self.loss_calc = nn.MSELoss(
                    reduction='none'
                )
        #if not self.beta:
        #    self.loss_calc = nn.SmoothL1Loss(
        #            reduction='none',
        #            delta=0.3
        #        )
        else:
            def neglogbeta(p, q, y):
                loss = torch.lgamma(p + q)
                loss -= torch.lgamma(p) + torch.lgamma(q)
                loss += (p - 1) * torch.log(y + EPS)
                loss += (q - 1) * torch.log(1 - y + EPS)
                return -loss 
            self.loss_calc = neglogbeta
            def beta_mean(p, q):
                return p / (p + q)
            self.beta_mean = beta_mean 
            def beta_std(p, q):
                return torch.sqrt(p*q / ((p+q)**2 * (p+q+1)))
            self.beta_std = beta_std
        self.reset()

    def reset(self):
        self.loss = 0 
        self.acc = 0 
        self.pos_acc = 0
        self.neg_acc = 0
        self.n_pos = 0
        self.n_particles = 0
        self.n_steps = 0
        self.hists = {}
        self.bins = {}

    @staticmethod
    def make_roc(pos_hist, neg_hist):
        pos_hist = pos_hist / pos_hist.sum()
        neg_hist = neg_hist / neg_hist.sum()
        tp, fp = [], []
        for i in np.arange(pos_hist.shape[0], -1, -1):
            tp.append(pos_hist[i:].sum())
            fp.append(neg_hist[i:].sum())
        auc = np.trapz(tp, x=fp)
        plt.plot(fp, tp, label=f'AUC={auc:.3f}')
        return fp, tp

    def add_values(self, val, key, w=None, lo=0, hi=1):
        hist, bins = np.histogram(
                val, bins=np.linspace(lo, hi, 100), weights=w)
        if key not in self.hists:
            self.hists[key] = hist + EPS
            self.bins[key] = bins
        else:
            self.hists[key] += hist

    def compute(self, yhat, y, w=None, m=None, plot_m=None):
        y = y.view(-1)
        if not self.beta:
            yhat = yhat.view(-1)
            loss = self.loss_calc(yhat, y)
        else:
            yhat = yhat + EPS
            p, q = yhat[:, :, 0], yhat[:, :, 1]
            p, q = p.view(-1), q.view(-1)
            loss = self.loss_calc(p, q, y)
            yhat = self.beta_mean(p, q)
            yhat_std = self.beta_std(p, q)
        yhat = torch.clamp(yhat, 0 , 1)
        if w is not None:
            wv = w.view(-1)
            loss *= wv
        if m is None:
            m = torch.ones_like(y, dtype=bool)
        if plot_m is None:
            plot_m = m
        m = m.view(-1).float()
        plot_m = m.view(-1)
        loss *= m

        nan_mask = t2n(torch.isnan(loss)).astype(bool)

        loss = torch.mean(loss)
        self.loss += t2n(loss).mean()

        if nan_mask.sum() > 0:
            yhat = t2n(yhat)
            print(nan_mask)
            print(yhat[nan_mask])
            if self.beta:
                p, q = t2n(p), t2n(q)
                print(p[nan_mask])
                print(q[nan_mask])
            print()

        plot_m = t2n(plot_m).astype(bool)
        y = t2n(y)[plot_m]
        if w is not None:
            w = t2n(w).reshape(-1)[plot_m]
        yhat = t2n(yhat)[plot_m]
        n_particles = plot_m.sum()

        # let's define positive/negative by >/< 0.5
        y_bin = y > 0.5 
        yhat_bin = yhat > 0.5

        acc = (y_bin == yhat_bin).sum() / n_particles
        self.acc += acc

        n_pos = y_bin.sum()
        pos_acc = (y_bin == yhat_bin)[y_bin].sum() / n_pos 
        self.pos_acc += pos_acc
        n_neg = (~y_bin).sum()
        neg_acc = (y_bin == yhat_bin)[~y_bin].sum() / n_neg 
        self.neg_acc += neg_acc

        self.n_pos += n_pos
        self.n_particles += n_particles

        self.n_steps += 1

        self.add_values(
            y, 'truth', w, -0.2, 1.2) 
        self.add_values(
            yhat, 'pred', w, -0.2, 1.2) 
        self.add_values(
            yhat-y, 'err', w, -2, 2)

        return loss, acc

    def mean(self):
        return ([x / self.n_steps 
                 for x in [self.loss, self.acc, self.pos_acc, self.neg_acc]]
                + [self.n_pos / self.n_particles])

    def plot(self, path):
        plt.clf()
        bins = self.bins['truth'] 
        x = (bins[:-1] + bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,
                'bins': bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists['truth'], label='Truth', **hist_args)
        plt.hist(weights=self.hists['pred'], label='Pred', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel(r'$E_{\mathrm{hard}}/E_{\mathrm{tot.}}$')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_e.' + ext)

        plt.clf()
        bins = self.bins['err'] 
        x = (bins[:-1] + bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,
                'bins': bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists['err'], label='Error', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel(r'Prediction - Truth')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_err.' + ext)


class PapuMetricsKL(object):
    def __init__(self, beta=False):
        self.beta = beta
        if not self.beta:
            self.loss_calc = nn.MSELoss(
                    reduction='none'
                )
        else:
            def neglogbeta(p, q, y):
                loss = torch.lgamma(p + q)
                loss -= torch.lgamma(p) + torch.lgamma(q)
                loss += (p - 1) * torch.log(y + EPS)
                loss += (q - 1) * torch.log(1 - y + EPS)
                return -loss 
            self.loss_calc = neglogbeta
            def beta_mean(p, q):
                return p / (p + q)
            self.beta_mean = beta_mean 
            def beta_std(p, q):
                return torch.sqrt(p*q / ((p+q)**2 * (p+q+1)))
            self.beta_std = beta_std
        self.reset()

    def reset(self):
        self.loss = 0 
        self.acc = 0 
        self.pos_acc = 0
        self.neg_acc = 0
        self.n_pos = 0
        self.n_particles = 0
        self.n_steps = 0
        self.hists = {}
        self.bins = {}

    @staticmethod
    def make_roc(pos_hist, neg_hist):
        pos_hist = pos_hist / pos_hist.sum()
        neg_hist = neg_hist / neg_hist.sum()
        tp, fp = [], []
        for i in np.arange(pos_hist.shape[0], -1, -1):
            tp.append(pos_hist[i:].sum())
            fp.append(neg_hist[i:].sum())
        auc = np.trapz(tp, x=fp)
        plt.plot(fp, tp, label=f'AUC={auc:.3f}')
        return fp, tp

    def add_values(self, val, key, w=None, lo=0, hi=1):
        hist, bins = np.histogram(
                val, bins=np.linspace(lo, hi, 100), weights=w)
        if key not in self.hists:
            self.hists[key] = hist + EPS
            self.bins[key] = bins
        else:
            self.hists[key] += hist

    def compute(self, yhat, y, pt, phi, neutral_mask, ybatch, genmet, genmetphi, w=None, m=None, plot_m=None):
        print('Computing met resolution')
        score = t2n(torch.clamp(yhat.squeeze(-1), 0, 1))
        charged_mask = ~neutral_mask
        score[charged_mask] = ybatch[charged_mask]
        y = y.view(-1)
        if not self.beta:
            yhat = yhat.view(-1)
            loss = self.loss_calc(yhat, y)
        else:
            yhat = yhat + EPS
            p, q = yhat[:, :, 0], yhat[:, :, 1]
            p, q = p.view(-1), q.view(-1)
            loss = self.loss_calc(p, q, y)
            yhat = self.beta_mean(p, q)
            yhat_std = self.beta_std(p, q)
        yhat = torch.clamp(yhat, 0 , 1)
        if w is not None:
            wv = w.view(-1)
            loss *= wv
        if m is None:
            m = torch.ones_like(y, dtype=bool)
        if plot_m is None:
            plot_m = m
        m = m.view(-1).float()
        plot_m = m.view(-1)
        loss *= m

        nan_mask = t2n(torch.isnan(loss)).astype(bool)

        loss = torch.mean(loss)

        randint = random.randint(0, 1000)
        if (randint%50==0):
            logger.info(f'RMS Loss: {t2n(loss).mean()}')
        self.loss += t2n(loss).mean()

        #compute MET
        pt = pt * score
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        ux = np.sum(px, axis=-1)
        uy = np.sum(py, axis=-1)
        u = np.sqrt(np.power(ux, 2) + np.power(uy, 2))
        uphi = np.arccos(ux/u)
        gmx = np.cos(genmetphi)
        gmy = np.sin(genmetphi)
        upar = []
        uper = []
        for i in range(len(ux)):
            upar.append(np.dot([ux[i],uy[i]],[gmx[i],gmy[i]]))
            if (u[i]*u[i] < upar[i]*upar[i]):
                upar[i] = u[i]*0.99999*(upar[i]/abs(upar[i]))
            uper.append(np.sqrt(u[i]*u[i] - upar[i]*upar[i]))
        upar = np.array(upar)

        #logger.info(upar)
        histu, _ = np.histogram((-1)*upar, bins=np.linspace(0, 220, 11), density=True)
        histgm, _ = np.histogram(genmet, bins=np.linspace(0, 220, 11), density=True)

        histu *= 22#*10/9 # binwidth
        histgm *= 22#*10/9 # binwidth

        #logger.info('WTF')
        #logger.info(histu)
        #logger.info(histgm)
        
        #collapse zero-entries into neighboring bins
        hist1,hist2 = collapsehist(histu,histgm)        
        #logger.info(hist1)
        histgm_final,histu_final = collapsehist(hist2,hist1)        

        #logger.info('HISTU')
        #logger.info(histu_final)
        #logger.info('HISTGM')
        #logger.info(histgm_final)
        #klloss = 0.001*nn.KLDivLoss(np.log(histu_final),histgm_final)
        klloss = 0.1*kl_divergence(torch.tensor(histu_final),torch.tensor(histgm_final))

        if (randint%50==0):
            logger.info(f'KL Loss: {klloss}')
        loss += klloss.float()
        self.loss += klloss.float()
        
        if nan_mask.sum() > 0:
            yhat = t2n(yhat)
            print(nan_mask)
            print(yhat[nan_mask])
            if self.beta:
                p, q = t2n(p), t2n(q)
                print(p[nan_mask])
                print(q[nan_mask])
            print()

        plot_m = t2n(plot_m).astype(bool)
        y = t2n(y)[plot_m]
        if w is not None:
            w = t2n(w).reshape(-1)[plot_m]
        yhat = t2n(yhat)[plot_m]
        n_particles = plot_m.sum()

        # let's define positive/negative by >/< 0.5
        y_bin = y > 0.5 
        yhat_bin = yhat > 0.5

        acc = (y_bin == yhat_bin).sum() / n_particles
        self.acc += acc

        n_pos = y_bin.sum()
        pos_acc = (y_bin == yhat_bin)[y_bin].sum() / n_pos 
        self.pos_acc += pos_acc
        n_neg = (~y_bin).sum()
        neg_acc = (y_bin == yhat_bin)[~y_bin].sum() / n_neg 
        self.neg_acc += neg_acc

        self.n_pos += n_pos
        self.n_particles += n_particles

        self.n_steps += 1

        self.add_values(
            y, 'truth', w, -0.2, 1.2) 
        self.add_values(
            yhat, 'pred', w, -0.2, 1.2) 
        self.add_values(
            yhat-y, 'err', w, -2, 2)
        
        return loss, acc

    def mean(self):
        return ([x / self.n_steps 
                 for x in [self.loss, self.acc, self.pos_acc, self.neg_acc]]
                + [self.n_pos / self.n_particles])

    def plot(self, path):
        plt.clf()
        bins = self.bins['truth'] 
        x = (bins[:-1] + bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,
                'bins': bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists['truth'], label='Truth', **hist_args)
        plt.hist(weights=self.hists['pred'], label='Pred', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel(r'$E_{\mathrm{hard}}/E_{\mathrm{tot.}}$')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_e.' + ext)

        plt.clf()
        bins = self.bins['err'] 
        x = (bins[:-1] + bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,
                'bins': bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists['err'], label='Error', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel(r'Prediction - Truth')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_err.' + ext)


class ParticleUResponse(METResolution):
    @staticmethod
    def _compute_res(pt, phi, w, gm, gmphi):
        pt = pt * w
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        ux = np.sum(px, axis=-1)
        uy = np.sum(py, axis=-1)
        u = np.sqrt(np.power(ux, 2) + np.power(uy, 2))
        uphi = np.arccos(ux/u)
        gmx = np.cos(gmphi)
        gmy = np.sin(gmphi)
        upar = []
        uper = []
        for i in range(len(ux)):
            upar.append(np.dot([ux[i],uy[i]],[gmx[i],gmy[i]]))
            if (u[i]*u[i] < upar[i]*upar[i]):
                upar[i] = u[i]*0.99999*(upar[i]/abs(upar[i]))
            uper.append(np.sqrt(u[i]*u[i] - upar[i]*upar[i]))
        upar = np.array(upar)
        uper = np.array(uper)
        #upar = np.dot([ux,uy],[gmx,gmy])
        #uper = np.sqrt(u*u - upar*upar)
        return u,uphi,ux,uy,gm,gmphi,gmx,gmy,upar,uper

    def compute(self, pt, phi, w, y, baseline, gm, gmphi):
        u_model,uphi_model,ux_model,uy_model,gm_model,gmphi_model,gmx_model,gmy_model,upar_model,uper_model = self._compute_res(pt, phi, w, gm, gmphi)
        u_truth,uphi_truth,ux_truth,uy_truth,gm_truth,gmphi_truth,gmx_truth,gmy_truth,upar_truth,uper_truth = self._compute_res(pt, phi, y, gm, gmphi)
        u_puppi,uphi_puppi,ux_puppi,uy_puppi,gm_puppi,gmphi_puppi,gmx_puppi,gmy_puppi,upar_puppi,uper_puppi = self._compute_res(pt, phi, baseline, gm, gmphi)
        u_pf,uphi_pf,ux_pf,uy_pf,gm_pf,gmphi_pf,gmx_pf,gmy_pf,upar_pf,uper_pf = self._compute_res(pt, phi, 1, gm, gmphi)

        bins = np.linspace(0., 300., num=25)

        df_model = pd.DataFrame()
        df_truth = pd.DataFrame()
        df_puppi = pd.DataFrame()
        df_pf = pd.DataFrame()
        df_model['upar'] = upar_model
        df_truth['upar'] = upar_truth
        df_puppi['upar'] = upar_puppi
        df_pf['upar'] = upar_pf
        df_model['uper'] = uper_model 
        df_truth['uper'] = uper_truth
        df_puppi['uper'] = uper_puppi
        df_pf['uper'] = uper_pf
        df_model['x'] = gm
        df_truth['x'] = gm
        df_puppi['x'] = gm
        df_pf['x'] = gm
        df_model['xphi'] = gmphi
        df_truth['xphi'] = gmphi
        df_puppi['xphi'] = gmphi
        df_pf['xphi'] = gmphi
        df_model['bin'] = np.searchsorted(bins, df_model['x'].values)
        df_truth['bin'] = np.searchsorted(bins, df_truth['x'].values)
        df_puppi['bin'] = np.searchsorted(bins, df_puppi['x'].values)
        df_pf['bin'] = np.searchsorted(bins, df_pf['x'].values)
        df_model['u'] = u_model
        df_truth['u'] = u_truth
        df_puppi['u'] = u_puppi
        df_pf['u'] = u_pf
        df_model['uphi'] = uphi_model
        df_truth['uphi'] = uphi_truth
        df_puppi['uphi'] = uphi_puppi
        df_pf['uphi'] = uphi_pf

        self.xedges = bins

        #print(res)
        #print(self.xedges)

        if self.df_model is None:
            self.df_model = df_model
            self.df_truth = df_truth
            self.df_puppi = df_puppi
            self.df_pf = df_pf
        else:
            tmp_model = pd.concat([self.df_model,df_model],ignore_index=True,sort=False)
            tmp_truth = pd.concat([self.df_truth,df_truth],ignore_index=True,sort=False)
            tmp_puppi = pd.concat([self.df_puppi,df_puppi],ignore_index=True,sort=False)
            tmp_pf = pd.concat([self.df_pf,df_pf],ignore_index=True,sort=False)
            self.df_model = tmp_model
            self.df_truth = tmp_truth
            self.df_puppi = tmp_puppi
            self.df_pf = tmp_pf

    def plot(self, path):

        self.df_pf.to_csv(path+'_pf.csv',index=False)
        self.df_model.to_csv(path+'_model.csv',index=False)
        self.df_truth.to_csv(path+'_truth.csv',index=False)
        self.df_puppi.to_csv(path+'_puppi.csv',index=False)

        plt.clf()

        #print(resp_df_binned)
        #print(self.xedges)
        upar_df_pf_binned = self.df_pf.groupby('bin', as_index=False)['upar'].mean()
        upar_df_model_binned = self.df_model.groupby('bin', as_index=False)['upar'].mean()
        upar_df_truth_binned = self.df_truth.groupby('bin', as_index=False)['upar'].mean()
        upar_df_puppi_binned = self.df_puppi.groupby('bin', as_index=False)['upar'].mean()
        genm_df_pf_binned = self.df_pf.groupby('bin', as_index=False)['x'].mean()
        genm_df_model_binned = self.df_model.groupby('bin', as_index=False)['x'].mean()
        genm_df_truth_binned = self.df_truth.groupby('bin', as_index=False)['x'].mean()
        genm_df_puppi_binned = self.df_puppi.groupby('bin', as_index=False)['x'].mean()

        resp_pf = np.array(upar_df_pf_binned['upar'].values)/np.array(genm_df_pf_binned['x'].values)
        resp_model = np.array(upar_df_model_binned['upar'].values)/np.array(genm_df_model_binned['x'].values)
        resp_truth = np.array(upar_df_truth_binned['upar'].values)/np.array(genm_df_truth_binned['x'].values)
        resp_puppi = np.array(upar_df_puppi_binned['upar'].values)/np.array(genm_df_puppi_binned['x'].values)

        print(resp_model)
        print(upar_df_model_binned)
        print(genm_df_model_binned)

        std_df_pf_binned = self.df_pf.groupby('bin', as_index=False)['uper'].std()
        std_df_model_binned = self.df_model.groupby('bin', as_index=False)['uper'].std()
        std_df_truth_binned = self.df_truth.groupby('bin', as_index=False)['uper'].std()
        std_df_puppi_binned = self.df_puppi.groupby('bin', as_index=False)['uper'].std()

        binid_to_genm = []
        for i in upar_df_model_binned['bin'].values:
            binid_to_genm.append(self.xedges[i-1])

        plt.clf()
        fig, ax = plt.subplots()
        plt.plot(binid_to_genm, (-1)*resp_pf, label = "PF", marker='o')
        plt.plot(binid_to_genm, (-1)*resp_model, label = "Model", marker='o')
        plt.plot(binid_to_genm, (-1)*resp_truth, label = "Truth", marker='o')
        plt.plot(binid_to_genm, (-1)*resp_puppi, label = "PUPPI", marker='o')
        plt.legend()
        plt.xlabel(r'Z $p_\mathrm{T}$ (GeV)')
        plt.ylabel(r'<$U_\mathrm{II}$>/<Z $p_\mathrm{T}$>')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_response.' + ext)


        plt.clf()
        fig, ax = plt.subplots()
        #plt.plot((self.xedges[1:] + self.xedges[:-1]) / 2, resp_df_binned, label = "Model", marker='o')
        #plt.plot((self.xedges[1:] + self.xedges[:-1]) / 2, resp_df_p_binned, label = "PUPPI", marker='o')
        plt.plot(binid_to_genm, np.array(std_df_pf_binned['uper'].values)/((-1)*resp_pf), label = "PF", marker='o')
        plt.plot(binid_to_genm, np.array(std_df_model_binned['uper'].values)/((-1)*resp_model), label = "Model", marker='o')
        plt.plot(binid_to_genm, np.array(std_df_truth_binned['uper'].values)/((-1)*resp_truth), label = "Truth", marker='o')
        plt.plot(binid_to_genm, np.array(std_df_puppi_binned['uper'].values)/((-1)*resp_puppi), label = "PUPPI", marker='o')
        plt.legend()
        plt.xlabel(r'Z $p_\mathrm{T}$ (GeV)')
        plt.ylabel(r'$\sigma(U_\mathrm{perp})$/(<$U_\mathrm{II}$>/<Z $p_\mathrm{T}$>)')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_perp.' + ext)

        return {'model': (1, np.sqrt(1)), 'puppi': (1, np.sqrt(1))}
