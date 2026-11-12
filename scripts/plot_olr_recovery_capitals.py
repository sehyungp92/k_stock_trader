"""Export the completed native capital diagnostic; never choose a policy."""
from pathlib import Path
import argparse
import json
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();study=json.loads((root/'study_100000000.json').read_text())
    sys.path.insert(0,study['code'])
    from backtests.strategies.olr.allocation_recovery import CAPITALS,SCENARIOS,checked_case,cell_failures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    names={'median':'Selected median control','native_auction_top2':'Bounded auction top-ups',
           'native_entry_recycle2':'Recycling: limited direct support'}
    colors={'median':'#4b5563','native_auction_top2':'#008774','native_entry_recycle2':'#d17b18'}
    rows={};summaries={}
    for name in names:
        for capital in CAPITALS:
            for scenario in SCENARIOS:
                case=root/'cases'/f'{name}_{capital}_{scenario}'
                result,_=checked_case(case)
                if result['source_generation_hash']!=study['source_generation_hash']:
                    raise ValueError('Capital plot mixes native execution generations')
                rows[name,capital,scenario]=result
    for name in names:
        failures=[];gains=[];gaps={}
        for capital in CAPITALS:
            for scenario in SCENARIOS:
                r=rows[name,capital,scenario];b=rows['median',capital,scenario]
                gains.append(r['return_pct']-b['return_pct'])
                if name!='median':
                    failures.extend(f'{capital}_{scenario}_{reason}' for reason in cell_failures(r,b,capital==100000000))
        for scenario in SCENARIOS:
            gap=rows[name,10000000,scenario]['return_pct']-rows[name,100000000,scenario]['return_pct']
            original=rows['median',10000000,scenario]['return_pct']-rows['median',100000000,scenario]['return_pct']
            gaps[scenario]={'return_gap_10m_minus_100m_pp':gap,'gap_reduction_pp':original-gap}
            if gap>original+1e-9:failures.append(scenario+'_wider_gap')
        summaries[name]={'minimum_per_cell_gain_pp':min(gains),'maximum_per_cell_gain_pp':max(gains),
            'economic_or_execution_failures':failures,'capital_gaps':gaps,
            'status':'Descriptive only; support and full source-audit requirements are evaluated by the registered framework'}
    export={'source_generation_hash':study['source_generation_hash'],'independent_evidence':False,
        'summaries':summaries,'cells':[{'profile':name,'capital':capital,'scenario':scenario,
            **{key:r[key] for key in ('return_pct','drawdown_pct','trades','execution_validity_pass','audit_status')}}
            for (name,capital,scenario),r in rows.items()]}
    (root/'capital_comparison.json').write_text(json.dumps(export,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
    capitals=[c/1000000 for c in CAPITALS]
    for column,scenario in enumerate(SCENARIOS):
        axes[0,column].set_title('Normal execution' if scenario=='baseline' else 'Combined execution stress',fontsize=12)
        for name,label in names.items():
            for row,metric in enumerate(('return_pct','drawdown_pct')):
                values=[rows[name,c,scenario][metric] for c in CAPITALS]
                axes[row,column].plot(capitals,values,marker='o',linewidth=2,color=colors[name],
                    linestyle='--' if name=='native_entry_recycle2' else '-',label=label)
        ceiling=[min(10.75,rows['median',c,scenario]['drawdown_pct']+1) for c in CAPITALS]
        axes[1,column].plot(capitals,ceiling,color='#b91c1c',linestyle=':',linewidth=1,label='Registered DD ceiling')
        for row in (0,1):
            axes[row,column].grid(alpha=.2);axes[row,column].yaxis.set_major_formatter(PercentFormatter(100))
            axes[row,column].spines[['top','right']].set_visible(False)
        axes[1,column].set_xlabel('Starting capital (KRW million)')
        axes[1,column].set_xticks(capitals)
    axes[0,0].set_ylabel('Net whole-account return')
    axes[1,0].set_ylabel('Maximum marked-equity drawdown')
    axes[0,0].legend(fontsize=8,loc='best');axes[1,1].legend(fontsize=8,loc='best')
    fig.suptitle('OLR allocation recovery across starting capital\n7 Aug 2025–31 Mar 2026 • reused development history',fontsize=14)
    fig.savefig(root/'capital_comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
