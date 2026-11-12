"""Read completed native evidence; never recollect timestamped results."""
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts import olr_participation_study_v5 as t
from scripts import olr_repair_report_v4 as diagnostic
from scripts.olr_opening_risk_report_v2 import p,h,table,num,cash
n,s,b=t.n,t.s,t.b
LABELS={'selected_control':'Retained selected','bounded_participation_rescue':'Bounded participation rescue'}


def compare(rows):
    return table(['Policy','IS return %','IS intraday DD %','IS entries','IS entry days',
        'OOS return %','OOS intraday DD %','OOS entries','OOS entry days'],
        [[label,*[v for w in ('train','oos') for v in (num(r[w]['return_pct']),num(r[w]['drawdown_pct']),r[w]['trades'],r[w]['entry_days'])]] for label,r in rows])


def chart(data):
    fig,axs=plt.subplots(2,2,figsize=(14,9))
    for i,row in enumerate(data['candidates']):
        for j,w in enumerate(('train','oos')):
            eq=n.read(Path(row['windows'][w]['folder'])/'equity.json')
            ts=np.array([np.datetime64(x[0][:19]) for x in eq]);nav=np.array([x[1] for x in eq])/1e8
            dd=100*(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1)
            axs[0,j].plot(ts,100*(nav-1),label=LABELS[row['label']],lw=1.4)
            axs[1,j].plot(ts,dd,lw=1.4)
    for j,title in enumerate(('IS: Aug 7, 2025–Mar 31, 2026','Inspected OOS: Apr 1–Jul 10, 2026')):
        axs[0,j].set_title(title);axs[0,j].set_ylabel('Cumulative net return (%)');axs[1,j].set_ylabel('Intraday drawdown (%)')
    for ax in axs.flat:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=2)
    fig.suptitle('OLR: bounded recovery of earlier-stage entry opportunities',fontsize=16)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(t.OUT/'results.png',dpi=160);plt.close(fig)


def main():
    data=n.read(t.OUT/'results.json');profile=n.read(t.OUT/'selected_research_policy.json');done=n.read(t.ROOT/'qualification_complete.json')
    assert profile['results_sha256']==done['results_sha256']==n.sha(t.OUT/'results.json')
    assert done['profile_sha256']==n.sha(t.OUT/'selected_research_policy.json')
    diagnostic.t=t;diag=diagnostic.diagnostics(data);n.write(t.OUT/'mechanism_diagnostics.json',diag)
    chart(data);control=data['candidates'][0];tested=data['candidates'][1]
    selected=next(r for r in data['candidates'] if r['id']==profile['selected_id'])
    reg=n.read(t.ROOT/'registration.json');preflight=n.read(t.OUT/'preflight_receipt.json')
    body='<h1>OLR: bounded participation-entry study</h1>'
    body+=p('<strong>Final research policy: '+escape(LABELS[selected['label']])+'. '+escape(profile['status'])+'</strong>')
    body+=p('This is one fixed earlier-stage entry treatment starting from the retained selected profile. It tests whether otherwise empty days contain usable stock-level alpha despite weak lagged sector participation. Neither thresholds nor gates were changed after outcomes. Faster-exit counterfactuals are not included in these native results.')
    body+='<img src="results.png" alt="IS and inspected OOS cumulative equity and intraday drawdown">'
    body+=compare([('Original',data['references']['original']),('Guard',data['references']['guard']['windows'])]+[(LABELS[r['label']],r['windows']) for r in data['candidates']])
    body+=p('Cumulative, nonannualized returns on separate KRW100m accounts. DD uses all native intraday marks, including unrealized profit retreat. Trade counts are unique filled entry lifecycles, with partial exit legs collapsed. April–July has been repeatedly inspected and is development data, not independent validation.')
    body+=h('Mechanism and funded opportunity')+p(escape(reg['mechanism']))+p(escape(reg['sizing']))
    body+=table(['Window','Confirmed added candidates','Six-bar signals','Signal days'],[[x['window'],x['confirmed_candidates'],x['six_bar_signals'],x['signal_days']] for x in preflight['evidence']])
    rows=[]
    for group in ('windows','stress','capped'):
        for w in ('train','oos'):
            a=tested[group][w]['authentication']['participation_envelope']
            e=__import__('scripts.olr_repair_qualify_v4',fromlist=['frequency_economics']).frequency_economics(tested[group][w],control[group][w])
            rows.append([group,w,a['submitted_orders'],a['filled_lifecycles'],cash(a['filled_entry_notional']),cash(a['net_pnl']),
                num(100*a['max_realized_entry_nav_fraction']),e['added'],e['removed'],cash(e['net_incremental_notional']),cash(e['median_added_notional'])])
    body+=table(['Scenario','Window','Submitted rescue orders','Filled rescue entries','Rescue notional','Rescue net P&L','Largest actual entry / admission NAV %','Added','Removed','Net additional notional','Median added notional'],rows)
    body+=p('5% per-name and 10% family limits are checked at admission using native buffered cost, marked family holdings and remaining pending reservations. They do not force later trims after appreciation. Actual fill notionals are reported separately because execution prices can differ. Pending exits provide no funding; the pilot receives no redistribution top-ups.')
    body+=h('Unchanged acceptance gates')
    for group,key in [('Nominal','gates'),('Combined stress','stress_gates'),('No upward compounding','capped_gates')]:
        body+=table([group+' gate','Passed'],[[k,v] for k,v in tested[key].items()])
    body+=p('Nominal IS floors: original−3pp, guard−3pp, selected−1.5pp; IS DD ≤ guard+2pp. OOS return ≥ selected+0.25pp and DD ≤ selected−0.75pp. At least 186/47 IS/OOS unique entries and 88/29 entry days. OOS median added notional and net incremental notional must each reach KRW1m. Stress retains all IS floors and requires OOS return at least control, lower DD, and no loss of entries or entry days. Capped sizing must also preserve IS within 1.5pp, improve OOS DD, preserve OOS return, and preserve both windows’ entries and entry days.')
    body+=h('Execution stress and compounding')
    body+=compare([(LABELS[r['label']]+' — combined stress',r['stress']) for r in data['candidates']])
    body+=p('Combined stress halves the declared broker participation from 1% to 0.5% and raises slippage from 5 to 15bps. The strategy does not anticipate the stressed fill quantities. Commission and tax remain native. Both policies are tested regardless of nominal success.')
    body+=compare([(LABELS[r['label']]+' — capped sizing',r['capped']) for r in data['candidates']])
    body+=p('No upward compounding caps the sizing base at starting capital after gains and lets it fall with losses. Every OOS comparison restores its own actual matching IS terminal history, including its stress or capped setting.')
    body+=h('Winner preservation and economic attribution')
    body+=table(['Window','Net P&L difference','Per-unit execution effect','Quantity effect','Added minus removed effect','All original winner P&L retained %','Removed winner P&L','Original winners becoming nonwinners'],
        [[w,cash(m['pnl_delta']),cash(m['per_unit_execution_effect']),cash(m['quantity_effect']),cash(m['new_minus_removed_effect']),
            num(diag[tested['id']][w]['all_original_winner_pnl_retention_pct']),cash(diag[tested['id']][w]['removed_winner_pnl']),m['winner_to_loser']] for w,m in tested['trade_comparison'].items()])
    body+=p('All-original-winner retention includes removed winners as zero and includes replacement losses. The P&L decomposition reconciles to total change, but is not a randomized causal estimate. Additional positions can change later cash, quantities and compounded exposure even though incumbent candidate selection is exactly preserved.')
    body+=table(['Window','Entry date','Symbol','Control quantity','Treatment quantity','Net P&L difference'],
        [[w,x['entry'][:10],x['symbol'],x['old_qty'],x['new_qty'],cash(x['pnl_delta'])] for w in ('train','oos') for x in diag[tested['id']][w]['economic_effects'][:8]])
    body+=table(['Policy','OOS peak NAV','OOS trough NAV','Peak-to-trough loss','DD %'],
        [[LABELS[r['label']],cash(r['windows']['oos']['intraday_drawdown']['peak_nav']),cash(r['windows']['oos']['intraday_drawdown']['trough_nav']),cash(r['windows']['oos']['intraday_drawdown']['peak_nav']-r['windows']['oos']['intraday_drawdown']['trough_nav']),num(r['windows']['oos']['drawdown_pct'])] for r in data['candidates']])
    body+=h('Warm-up continuity and limits')
    body+=p('The August 7 IS start comes from the inherited 60-session training plus one-session embargo. It is not the start of historical market data. Before April 1, each policy/scenario/capped replay restores its actual March 31 terminal market-history state. Each historical sample is verified against independently retained OOS OHLCV. The flat account resets capital; market history does not reset. The 217 prior source sessions do not create auction observations that the routed strategy never observed.')
    body+=table(['Policy','IS entries with sparse auction history','OOS entries with sparse auction history'],[[LABELS[r['label']],diag[r['id']]['train']['sparse_auction_trades'],diag[r['id']]['oos']['sparse_auction_trades']] for r in data['candidates']])
    body+=p('Fixed universe/taxonomy, sparse auction history and five-minute price/volume proxies remain. A successful historical result would still need new source-qualified validation. This pilot does not prove optimality or future expected returns. One-minute exit execution is a separate engineering and qualification task; the earlier fixed-entry proceeds screen cannot be counted as strategy P&L or DD reduction.')
    body+=p('Files: <a href="selected_research_policy.json">selected research profile</a> · <a href="registration.json">frozen registration</a> · <a href="review_bundle/implementation.patch">implementation patch</a> · <a href="mechanism_diagnostics.json">trade diagnostics</a>. No live policy was activated.')
    html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OLR bounded participation study</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:40px auto;padding:0 25px;color:#17243a}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}th,td{border:1px solid #ccd5df;padding:7px;text-align:left}th{background:#eaf0f7}img{width:100%;height:auto}h1,h2{line-height:1.25}a{color:#174e98}</style>'+body+'</html>'
    (t.OUT/'report.html').write_text(html,encoding='utf-8')
    class Extract(HTMLParser):
        def __init__(self):super().__init__();self.parts=[]
        def handle_data(self,v):self.parts.append(v.strip())
        def handle_starttag(self,tag,attrs):
            if tag in ('p','h1','h2','tr','td','th'):self.parts.append('\n' if tag not in ('td','th') else ' | ')
    parser=Extract();parser.feed(body)
    (t.OUT/'report.txt').write_text(''.join(parser.parts),encoding='utf-8')
    n.write(t.OUT/'report_receipt.json',dict(results_sha256=profile['results_sha256'],profile_sha256=done['profile_sha256'],
        files={name:n.sha(t.OUT/name) for name in ('report.html','report.txt','results.png','mechanism_diagnostics.json')},script_sha256=n.sha(__file__)))
    print(str(t.OUT/'report.html'))


if __name__=='__main__':main()
