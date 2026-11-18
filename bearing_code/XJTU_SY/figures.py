"""Render exactly the four requested scientific tables/figures from measured outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
COLORS=['#2166ac','#e08214','#b2182b']

def save(fig,out,name):
    for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def draw_table(headers,rows,title,footnote,out,name,widths=None):
    fig,ax=plt.subplots(figsize=(12,max(3,0.47*len(rows)+1.6)))
    ax.axis('off');ax.set_title(title,loc='left',fontweight='bold',pad=18)
    tbl=ax.table(cellText=rows,colLabels=headers,cellLoc='left',colLoc='left',loc='upper center',colWidths=widths)
    tbl.auto_set_font_size(False);tbl.set_fontsize(9);tbl.scale(1,2.1)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor('#d5dce3');cell.PAD=.055
        if r==0:cell.set_facecolor('#23455d');cell.set_text_props(color='white',weight='bold')
        else:cell.set_facecolor('#f0f4f7' if r%2 else 'white')
    fig.text(.125,.025,footnote,fontsize=9,va='bottom')
    save(fig,out,name)

def settings_table(cfg,out,manifest):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for c in [1,2,3]:
        counts=[sum(manifest['bearings'][b]['records'] for b in cfg['split'][s] if b.startswith(f'Bearing{c}_')) for s in ['train','val','test']]
        rows.append([f'Condition {c}',f'{c}_1 / {c}_2 / {c}_3 | {c}_4 | {c}_5',f'{counts[0]} / {counts[1]} / {counts[2]} records'])
    rows += [
        ['Signal','Horizontal vibration',f'{cfg["sample_rate_hz"]:,} Hz; 1.28 s every 1 min'],
        ['Short windows',f'{cfg["window_points"]} points, no overlap','32 windows per acquisition record'],
        ['Preprocessing','DC removal -> Hann -> FFT','Train-only global normalization per domain'],
        ['Historical input',f'{cfg["history"]} acquisition records','Past/current observations only'],
        ['Architecture','CNN + hypergraph + parallel attention + BiLSTM','Separate stage and RUL models'],
        ['Graph / sequence','200 -> 128; 4 GAT heads; BiLSTM 128/direction',f'Cosine threshold tau = {cfg["tau"]}'],
        ['Proxy stages','Trailing RMS median; initial 10-record baseline','z < 3; 3 <= z < 6; z >= 6'],
        ['RUL target','Minutes until final recorded endpoint','No test lifetime used for input normalization'],
        ['Training',f'Adam; lr={cfg["learning_rate"]}; up to {cfg["epochs"]} epochs',f'Early stopping {cfg["patience"]}; seeds={cfg["seeds"]}'],
        ['Selection','Stage: validation Macro-F1; RUL: validation MAE','Independent bearing-level train/val/test split']]
    pd.DataFrame(rows,columns=['Setting','Specification','Details']).to_csv(out/'01_experiment_settings.csv',index=False,encoding='utf-8-sig')
    draw_table(['Setting','Specification / bearing IDs (train | val | test)','Details / record counts (train / val / test)'],rows,
         'XJTU-SY: data split and experimental settings',
         'Bearing IDs omit the prefix "Bearing". Stages are vibration-derived proxies, not measured physical damage grades.',out,'01_experiment_settings',[.15,.44,.41])

def make_figures(cfg,out):
    out=Path(out);m=pd.read_csv(out/'test_metrics_by_seed.csv');nseed=len(cfg['seeds'])
    grouped=m.groupby('bearing',sort=False);rows=[];summary=[]
    for bid in cfg['split']['test']:
        f=grouped.get_group(bid);r={'bearing':bid,'n_predictions':int(f.n_predictions.iloc[0])}
        for col in ['accuracy','macro_f1','rul_mae_minutes']:
            r[col+'_mean']=float(f[col].mean());r[col+'_sd']=float(f[col].std(ddof=1)) if nseed>1 else None
        summary.append(r)
        def fmt(col,pct=False):
            scale=100 if pct else 1;v=f[col]*scale
            return f'{v.mean():.2f}'+(f' +/- {v.std(ddof=1):.2f}' if nseed>1 else '')
        support='/'.join(str(int(f[f'stage{k}_support'].iloc[0])) for k in range(3))
        rows.append([bid,str(r['n_predictions']),support,fmt('accuracy',True),fmt('macro_f1',True),fmt('rul_mae_minutes')])
    means=m.groupby('seed')[['accuracy','macro_f1','rul_mae_minutes']].mean()
    def avg(col,pct=False):
        v=means[col]*(100 if pct else 1)
        return f'{v.mean():.2f}'+(f' +/- {v.std(ddof=1):.2f}' if nseed>1 else '')
    rows.append(['Bearing mean','--','--',avg('accuracy',True),avg('macro_f1',True),avg('rul_mae_minutes')])
    pd.DataFrame(summary).to_csv(out/'02_test_summary.csv',index=False,encoding='utf-8-sig')
    note=('Single-seed exploratory results (seed '+str(cfg['seeds'][0])+').' if nseed==1 else f'Mean +/- sample SD across {nseed} seeds.')
    draw_table(['Test bearing','Predictions','Stage support\n0 / 1 / 2','Accuracy (%)','Macro-F1 (%)','RUL MAE (min)'],rows,
        'XJTU-SY: proxy-stage classification and RUL prediction',note+' Macro-F1 uses all three proxy classes (absent classes contribute zero).\nInitial baseline records excluded. Final row weights bearings equally. Proxy labels do not measure physical damage.',
        out,'02_test_summary',[.19,.12,.18,.16,.16,.19])
    # Fixed first seed selected before testing; never choose a visually favorable run.
    seed=cfg['seeds'][0];fig,axes=plt.subplots(3,1,figsize=(11,8),constrained_layout=True)
    for ax,bid in zip(axes,cfg['split']['test']):
        f=pd.read_csv(out/f'seed_{seed}'/(bid+'_predictions.csv'))
        ax.step(f.elapsed_minutes,f.proxy_stage,where='post',color='#222222',lw=1.8,label='RMS-derived reference')
        ax.step(f.elapsed_minutes,f.predicted_stage,where='post',color='#2166ac',lw=1.4,ls='--',label='Model prediction')
        ax.set_yticks([0,1,2],['Baseline','Degraded','Strongly degraded']);ax.set_ylim(-.2,2.25)
        ax.set_title(bid,loc='left',fontweight='bold');ax.set_xlabel('Elapsed time (min)');ax.grid(alpha=.2)
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.34),ncol=2,frameon=False)
    fig.suptitle(f'Proxy-stage recognition on independent test bearings (seed {seed})\nVibration-defined stages; not physical damage ground truth',fontsize=13)
    save(fig,out,'03_stage_recognition')
    fig,axes=plt.subplots(3,1,figsize=(11,8),constrained_layout=True)
    for ax,bid in zip(axes,cfg['split']['test']):
        f=pd.read_csv(out/f'seed_{seed}'/(bid+'_predictions.csv'))
        mae=float(np.abs(f.rul_minutes-f.predicted_rul_minutes).mean())
        ax.plot(f.elapsed_minutes,f.rul_minutes,color='#222222',lw=2,label='Reference RUL')
        ax.plot(f.elapsed_minutes,f.predicted_rul_minutes,color='#b2182b',lw=1.5,label='Predicted RUL')
        ax.set_title(f'{bid}   |   MAE = {mae:.2f} min',loc='left',fontweight='bold')
        ax.set_xlabel('Elapsed time (min)');ax.set_ylabel('RUL (min)');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.suptitle(f'RUL prediction on independent test bearings (seed {seed})\nInputs contain only the current and previous observations',fontsize=13)
    save(fig,out,'04_rul_prediction')
    print('FOUR TABLES/FIGURES EXPORTED',flush=True)
