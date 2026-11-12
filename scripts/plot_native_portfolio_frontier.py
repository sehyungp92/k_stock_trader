"""Render registered native portfolio search results without changing selection."""
import argparse
import json
from pathlib import Path


def plot(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    f=json.loads((root/'finalist.json').read_text());results=f['results']
    kvals=(50,75,100);ovals=(35,50,75,100)
    data=[]
    for scenario in (0,1):
        for recycle in (0,1):
            data.append(np.asarray([[results[f'k{k:03}_o{o:03}_r{recycle}'][scenario]['return_pct'] for o in ovals] for k in kvals]))
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for ax,values,(scenario,recycle) in zip(axes.flat,data,((s,r) for s in (0,1) for r in (0,1))):
        ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=max(x.max() for x in data)*1.15,aspect='auto')
        for i,k in enumerate(kvals):
            for j,o in enumerate(ovals):
                name=f'k{k:03}_o{o:03}_r{recycle}';r=results[name][scenario]
                valid=name in f['feasible'];foreground='white' if values[i,j]>max(x.max() for x in data)*.62 else '#122738'
                ax.text(j,i,f"Return {r['return_pct']:.1f}%\nMax DD {r['drawdown_pct']:.2f}%\n{r['trades']} entries",
                        ha='center',va='center',fontsize=10,color=foreground,linespacing=1.35)
                if valid:ax.add_patch(Rectangle((j-.47,i-.46),.94,.92,fill=False,edgecolor='#39c77d',linewidth=2.5))
                if name==f['winner']:ax.add_patch(Rectangle((j-.49,i-.49),.98,.98,fill=False,edgecolor='#f4a300',linewidth=4))
        ax.set_xticks(range(4),['35%','50%','75%','100%']);ax.set_yticks(range(3),['0.50×','0.75×','1.00×'])
        ax.set_xlabel('OLR virtual sizing equity and book budget / portfolio NAV')
        ax.set_ylabel('KALCB native sizing multiplier')
        ax.set_title(('Baseline' if scenario==0 else 'Combined cost + capacity stress')+' · Recycling '+('on' if recycle else 'off'),fontsize=12)
    fig.suptitle('KRW25m native portfolio budget search\n2025-08-07 – 2026-03-31 · Net cumulative returns · Reused development data',fontsize=16)
    fig.supxlabel('Green outline: passes both risk, return and frequency gates. Gold outline: base-grid selection.\nMigration and standalone controls are included in the accompanying result file.',fontsize=10)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plot(args.root,args.output)
