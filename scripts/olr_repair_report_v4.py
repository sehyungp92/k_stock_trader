"""Report completed v4 evidence; never regenerate selected result identities."""
from collections import Counter
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import gzip,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts import olr_repair_study_v4 as t
from scripts.olr_opening_risk_report_v2 import p,h,table,num,cash
n,s,b=t.n,t.s,t.b
LABELS={'selected_control':'Retained selected control','structural_partial_half':'Existing selective half exit',
 'quantity_consistent_tail':'Quantity-consistent legacy tail','separated_peer_price_tail':'Own liquidity + peer price',
 'aligned_portfolio_shock':'Aligned portfolio stress cap','empty_afternoon_flow_rescue':'Empty-afternoon flow rescue',
 'registered_component_union':'Registered component union'}

def compare(policies):
    return table(['Policy','IS return %','IS DD %','IS entries','IS entry days','OOS return %','OOS DD %','OOS entries','OOS entry days'],
        [[label,*[v for w in ('train','oos') for v in (num(windows[w]['return_pct']),num(windows[w]['drawdown_pct']),windows[w]['trades'],windows[w]['entry_days'])]] for label,windows in policies])

def diagnostics(data):
    output={}
    key=lambda x:(x['symbol'],x['entry_decision_time'])
    controls={w:{key(x):x for x in n.read(Path(data['candidates'][0]['windows'][w]['folder'])/'trades.json')} for w in ('train','oos')}
    for row in data['candidates']:
        output[row['id']]={}
        for w in ('train','oos'):
            folder=Path(row['windows'][w]['folder']);trades=n.read(folder/'trades.json');counts=Counter();records=[];partial_orders=0;partial_fills=0
            with gzip.open(folder/'native_decisions.jsonl.gz','rt',encoding='utf-8') as f:
                for line in f:
                    ev=json.loads(line)
                    if ev.get('decision_code') in ('ENTRY_SKIPPED','ENTRY_SUBMITTED'):counts[ev.get('reason','unknown')]+=1
                    if ev.get('reason')=='weak_followthrough_partial':
                        partial_orders+=ev.get('decision_code')=='EXIT_SUBMITTED'
                        partial_fills+=ev.get('decision_code')=='EXIT_FILLED'
            for tr in trades:
                m=tr['route_metadata']
                if m.get('risk_repair_evidence') or m.get('empty_sector_flow_rescue_v4'):
                    records.append(dict(symbol=tr['symbol'],entry=tr['entry_decision_time'],qty=tr['qty'],notional=tr['qty']*tr['entry_price'],net_pnl=tr['net_pnl'],
                        rescued=bool(m.get('empty_sector_flow_rescue_v4')),risk=m.get('risk_repair_evidence'),tail=m.get('liquidation_tail_evidence')))
            original=controls[w];current={key(x):x for x in trades}
            effects=[dict(symbol=k[0],entry=k[1],old_qty=original.get(k,{}).get('qty',0),new_qty=current.get(k,{}).get('qty',0),
                old_pnl=original.get(k,{}).get('net_pnl',0.),new_pnl=current.get(k,{}).get('net_pnl',0.),
                pnl_delta=current.get(k,{}).get('net_pnl',0.)-original.get(k,{}).get('net_pnl',0.)) for k in set(original)|set(current)]
            winners={k:x for k,x in original.items() if x['net_pnl']>0}
            winner_total=sum(x['net_pnl'] for x in winners.values())
            output[row['id']][w]=dict(entry_decision_reasons=dict(counts),mechanism_trades=records,
                selective_partial_orders=partial_orders,selective_partial_fill_events=partial_fills,
                economic_effects=sorted(effects,key=lambda x:(-abs(x['pnl_delta']),x['entry'],x['symbol'])),
                matched_quantity_changes=sum(x['qty']!=current[k]['qty'] for k,x in original.items() if k in current),
                all_original_winner_pnl_retention_pct=100*sum(current.get(k,{}).get('net_pnl',0.) for k in winners)/winner_total if winner_total else None,
                removed_winner_pnl=sum(x['net_pnl'] for k,x in winners.items() if k not in current),
                sparse_auction_trades=sum(x['route_metadata']['capacity_forecast']['exit']['status']=='insufficient_history' for x in trades))
    return output

def chart(data):
    fig,axes=plt.subplots(2,2,figsize=(15,10));colors=plt.get_cmap('tab10').colors
    for i,row in enumerate(data['candidates']):
        for j,w in enumerate(('train','oos')):
            eq=n.read(Path(row['windows'][w]['folder'])/'equity.json');dates=np.array([np.datetime64(x[0][:19]) for x in eq]);values=np.array([x[1] for x in eq])/1e8
            dd=100*(values/np.maximum.accumulate(np.r_[1.,values])[1:]-1)
            axes[0,j].plot(dates,100*(values-1),color=colors[i],label=LABELS[row['label']],lw=1.2)
            axes[1,j].plot(dates,dd,color=colors[i],lw=1.2)
    for j,title in enumerate(('IS: Aug 7, 2025–Mar 31, 2026','Inspected OOS: Apr 1–Jul 10, 2026')):
        axes[0,j].set_title(title);axes[0,j].set_ylabel('Cumulative net return (%)');axes[1,j].set_ylabel('Intraday drawdown (%)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=9)
    fig.suptitle('OLR: risk-model repairs and conditional opportunity recovery',fontsize=17);fig.tight_layout(rect=(0,.17,1,.95))
    fig.savefig(t.OUT/'results.png',dpi=160,bbox_inches='tight');plt.close(fig)

def main():
    t.activate();data=n.read(t.OUT/'results.json');profile=n.read(t.OUT/'selected_research_policy.json');done=n.read(t.ROOT/'qualification_complete.json')
    assert profile['results_sha256']==done['results_sha256']==n.sha(t.OUT/'results.json')
    assert done['profile_sha256']==n.sha(t.OUT/'selected_research_policy.json')
    diag=diagnostics(data);n.write(t.OUT/'mechanism_diagnostics.json',diag);chart(data)
    control=data['candidates'][0];selected=next(r for r in data['candidates'] if r['id']==profile['selected_id']);reg=n.read(t.ROOT/'registration.json')
    body='<h1>OLR: targeted risk repairs and conditional entry recovery</h1>'
    body+=p('<strong>'+('Qualified research successor: ' if selected['id']!=control['id'] else 'Final retained policy: ')+escape(LABELS[selected['label']])+'</strong>')
    body+=p('This registered study starts from the retained v3 profile. It separates two liquidation-estimator repairs, one admission-time portfolio stress overlay, the existing selective exit, and a conditional entry source on otherwise empty afternoons. Final promotion requires every return, DD, frequency and execution-stress gate. Component stress evidence alone does not qualify a successor. No live policy was activated; April–July remains repeatedly inspected development data.')
    body+='<img src="results.png" alt="Native IS and OOS equity and intraday drawdown comparisons">'
    body+=compare([('Original',data['references']['original']),('Guard',data['references']['guard']['windows'])]+[(LABELS[r['label']],r['windows']) for r in data['candidates']])
    body+=p('Returns are cumulative on separate flat-start KRW 100m accounts, not annualized. DD includes all native intraday marks and unrealized profit retreat. IS has 156 sessions; OOS has 69 including runoff. Unique entries collapse partial exit legs; entry days require actual first fills. The period boundary resets account capital, not market history.')
    body+=h('Registered mechanisms and what actually changed')
    for label,description in reg['hypotheses'].items():body+=p('<strong>'+escape(LABELS[label])+':</strong> '+escape(description))
    body+=p('The portfolio overlay uses the maximum supported common-date scenario, deliberately distinct from the two ES90 sizing repairs. It only limits new admissions/top-ups. It does not trim existing holdings, assume an earlier opening fill, or claim to capture all unseen gaps. The original tail panel already included the first capacity-eligible 09:05 price.')
    rows=[]
    for r in data['candidates'][1:]:
        for w in ('train','oos'):
            m=diag[r['id']][w]['mechanism_trades'];rescued=[x for x in m if x['rescued']];e=r['frequency_economics'][w]
            rows.append([LABELS[r['label']],w,len(m),len(rescued),cash(sum(x['notional'] for x in rescued)),cash(sum(x['net_pnl'] for x in rescued)),e['added'],e['removed'],cash(e['net_incremental_notional']),cash(e['median_added_notional'])])
    body+=table(['Policy','Window','Risk/rescue evidence entries','Rescued entries','Rescued notional','Rescued P&L','Added','Removed','Net added notional','Median added notional'],rows)
    body+=p('Risk/rescue evidence entries indicate attached entry metadata, not necessarily a changed size; the selective exit has separate activation counts below. Native decision reasons and quantity/risk proofs are exported in mechanism_diagnostics.json and each case’s native_decisions.jsonl.gz. Rescued candidates must still pass entry confirmation, real cash, risk and execution capacity. The rescue changes no incumbent candidate rank/features on originally nonempty days; preflight validates this over both source archives. Its opportunity source was limited to four stage-local empty OOS days before further conditions.')
    body+=table(['Policy','Window','Selective partial-exit orders','Partial fill events'],[[LABELS[r['label']],w,diag[r['id']][w]['selective_partial_orders'],diag[r['id']][w]['selective_partial_fill_events']] for r in data['candidates'] if r['mutations']['capacity_structural_partial'] for w in ('train','oos')])
    body+=p('Orders and partial fill events are execution activity, not additional entries or independent trades.')
    body+=h('Qualification and component robustness')
    body+=table(['Policy','Nominal qualified','Failed nominal gates','Stress tested','Stress qualified','Failed stress gates'],[[LABELS[r['label']],r.get('nominal_qualified') if r['id']!=control['id'] else 'Reference',', '.join(k for k,v in r.get('gates',{}).items() if not v) if r['id']!=control['id'] else 'Successor-improvement gates do not qualify the reference against itself',bool(r['stress']),r.get('stress_qualified','not run') if r['id']!=control['id'] else 'Reference',', '.join(k for k,v in r.get('stress_gates',{}).items() if not v) if r['id']!=control['id'] else 'Reference'] for r in data['candidates']])
    body+=p('IS return floors remain original−3pp, guard−3pp and selected−1.5pp, with IS DD ≤ guard+2pp. Nominal OOS return must improve ≥0.25pp and DD fall ≥0.75pp; entries must reach 186/47 and entry days 88/29. Added OOS entries must also have ≥KRW 1m median notional and ≥KRW 1m net incremental entry notional. These are economic frequency checks, not evidence of independent alpha.')
    body+=compare([(LABELS[r['label']]+' — combined stress',r['stress']) for r in data['candidates'] if len(r['stress'])==2])
    body+=p('Stress halves broker participation from 1% to 0.5% and raises OLR slippage from 5 to 15 bps under the inherited assumptions. Strategy capacity estimates retain the configured participation; adverse execution is not assumed known in advance. Every stressed OOS run restores the actual matching stressed IS terminal history. Control and the existing selective exit are always tested; additional component tests are selected by the registered rule, without requiring a risk-only component to create new trades.')
    body+=compare([(LABELS[r['label']]+' — no upward compounding',r['capped']) for r in data['candidates'] if len(r['capped'])==2])
    body+=p('Capped sizing limits the sizing base to starting capital when NAV rises while permitting it to fall with losses. Cash remains the real native account. This comparison measures sensitivity to reinvesting profits; it is not a substitute for passing the original IS gates.')
    body+=h('Winner preservation, displacement and concentration')
    body+=table(['Policy','Window','P&L delta','Per-unit effect','Quantity effect','Added−removed effect','Matched winner retained %','Top-five winner retained %','Removed entries','Winner→nonwinner'],
      [[LABELS[r['label']],w,cash(m['pnl_delta']),cash(m['per_unit_execution_effect']),cash(m['quantity_effect']),cash(m['new_minus_removed_effect']),num(m['matched_winner_pnl_retention_pct']),num(m['top_five_original_winner_pnl_retention_pct']),m['removed_count'],m['winner_to_loser']] for r in data['candidates'][1:] for w,m in r['trade_comparison'].items()])
    body+=p('The three P&L effects reconcile exactly to total change but are accounting decompositions, not separate randomized causal estimates. Matched-winner retention excludes removed winners; read it with membership changes. For the selective exit, the previous OOS saving was concentrated in one 034730 event; stress/capped comparisons now test that component without changing its trigger.')
    body+=table(['Policy','Window','All original winners retained %','Removed winner P&L','Matched quantities changed'],
        [[LABELS[r['label']],w,num(diag[r['id']][w]['all_original_winner_pnl_retention_pct']),cash(diag[r['id']][w]['removed_winner_pnl']),diag[r['id']][w]['matched_quantity_changes']] for r in data['candidates'][1:] for w in ('train','oos')])
    body+=p('The all-original-winner measure includes removed winners with zero replacement P&L and includes losses when an original winner becomes a loser. It therefore avoids the survivor bias of reporting matched winners alone. Quantity changes can reflect later cash and compounding effects as well as direct risk clipping.')
    body+=h('Trade-level findings and why the mechanisms did not combine into a full improvement')
    body+=p('Correcting risk arithmetic is necessary, but it does not itself make a historical risk estimate discriminate between future winners and losers. The quantity-only repair still transfers peer full-notional liquidity impairment and clips profitable exposure. Separating peer price risk largely removes that damage, but changes little of the OOS exposure that causes the drawdown. The aligned portfolio maximum-scenario cap reaches more exposure, including large winners. It improves DD at an excessive return and frequency cost under the registered gates.')
    body+=table(['Policy','OOS marked peak','OOS marked trough','Peak-to-trough KRW loss','DD %'],
        [[LABELS[r['label']],cash(r['windows']['oos']['intraday_drawdown']['peak_nav']),cash(r['windows']['oos']['intraday_drawdown']['trough_nav']),cash(r['windows']['oos']['intraday_drawdown']['peak_nav']-r['windows']['oos']['intraday_drawdown']['trough_nav']),num(r['windows']['oos']['drawdown_pct'])] for r in data['candidates']])
    body+=p('All nominal variants share the May 15 09:00 marked peak and July 9 12:15 trough. The portfolio cap lowers both levels: roughly KRW 5.31m less peak wealth and KRW 3.31m less trough wealth than control. Its lower percentage drawdown therefore does not mean it preserves more wealth at the trough. The half exit raises both endpoints; much of its saving occurs before the peak, explaining why a positive return gain translates into only a small DD reduction.')
    body+=table(['Policy','Window','Entry','Symbol','Control qty','Test qty','P&L difference'],
        [[LABELS[r['label']],w,x['entry'][:10],x['symbol'],x['old_qty'],x['new_qty'],cash(x['pnl_delta'])] for r in data['candidates'][1:] for w in ('train','oos') for x in [z for z in diag[r['id']][w]['economic_effects'] if abs(z['pnl_delta'])>1e-6][:3]])
    dominant=sorted(n.read(Path(control['windows']['oos']['folder'])/'trades.json'),key=lambda x:x['net_pnl'])[:4]
    bylabel={r['label']:r for r in data['candidates']}
    body+=table(['Control OOS loss entry','Control qty','Control net P&L','Peer-price repair qty','Portfolio cap qty','Half-exit P&L difference'],
        [[x['symbol']+' '+x['entry_decision_time'][:10],x['qty'],cash(x['net_pnl']),
          next(z['new_qty'] for z in diag[bylabel['separated_peer_price_tail']['id']]['oos']['economic_effects'] if z['symbol']==x['symbol'] and z['entry']==x['entry_decision_time']),
          next(z['new_qty'] for z in diag[bylabel['aligned_portfolio_shock']['id']]['oos']['economic_effects'] if z['symbol']==x['symbol'] and z['entry']==x['entry_decision_time']),
          cash(next(z['pnl_delta'] for z in diag[bylabel['structural_partial_half']['id']]['oos']['economic_effects'] if z['symbol']==x['symbol'] and z['entry']==x['entry_decision_time']))] for x in dominant])
    body+=p('These are the four largest realized OOS losses, not a complete attribution of the peak-to-trough marked drawdown. Winner givebacks and the timing of the peak also matter. The half exit preserves OOS winner exits, yet one 034730 saving exceeds its aggregate net gain. Its IS advantage also shrinks or reverses under stress/capped sizing. Combining it with the peer-price repair inherits those limits rather than adding an independent opportunity source.')
    rescue_review=n.read(t.OUT/'rescue_signal_review.json')
    body+=table(['Window','Candidate date','Symbol','Six-bar confirmation','Actual entries','Actual entry notional','Net P&L'],
        [[x['window'],x['date'],x['symbol'],x['six_bar_signal_time'] or 'None',x['actual_entries'],cash(x['actual_notional']),cash(x['actual_net_pnl'])] for x in rescue_review['rows']])
    body+=p('The rescue produced one actual IS trade, 042700 on March 6: KRW 55.97m notional and a KRW 5.41m loss. All incumbent IS trades retained exactly the same P&L and quantities. The other IS candidate and the sole OOS candidate did not pass the unchanged six-bar confirmation rule, so their failure to trade is not a cash or execution-capacity rejection. This reconstruction uses frozen source bars and does not extend the entry window or invent fills. The experiment does not support relaxing the sector-flow gate on this basis.')
    body+=table(['Policy','OOS month','Return %','Difference from control, pp'],[[LABELS[r['label']],month,num(v['return_pct']),num(v['return_pct']-control['windows']['oos']['months'][month]['return_pct'])] for r in data['candidates'][1:] for month,v in r['windows']['oos']['months'].items()])
    body+=h('History, execution and remaining limits')
    body+=p('IS starts August 7 because of the inherited 60-session training plus one-session embargo schedule. Every April 1 run restores its own actual March 31 market-history checkpoint, matched by candidate, stress and capped mode. Tail-enabled runs also restore the actual liquidation panel and match the independently sourced OOS prefix exactly. Older intraday data remain available. The 217 prior OOS source sessions do not eliminate sparse routed auction observations.')
    body+=table(['Policy','IS sparse auction entries','OOS sparse auction entries'],[[LABELS[r['label']],diag[r['id']]['train']['sparse_auction_trades'],diag[r['id']]['oos']['sparse_auction_trades']] for r in data['candidates']])
    body+=p('Fixed universe/taxonomy, sparse auction history, conservative unsupported-liquidity treatment and five-minute price/volume proxies remain. There is no order-book replay or guaranteed stop loss. Missing common dates reduce the portfolio overlay’s support; at least twenty aligned dates are required for new risk. The legacy sector-volume feature and score calibration are unchanged. No claim of future expected-return improvement is justified without new source-qualified validation.')
    body+=p('Final policy: <strong>'+escape(LABELS[selected['label']])+'</strong>. '+('It passed all nominal and stress gates; compare capped sensitivity and the detailed trade effects before any separate deployment decision.' if selected['id']!=control['id'] else 'No tested alternative passed every final gate. The component tables show any partial improvements and their remaining failures without changing the acceptance criteria.'))
    body+=p('<a href="selected_research_policy.json">Selected research profile</a> · <a href="results.json">Full results</a> · <a href="qualified_alternatives.json">Qualified alternatives</a> · <a href="mechanism_diagnostics.json">Mechanism evidence</a> · <a href="rescue_signal_review.json">Rescue signal reconstruction</a> · <a href="registration.json">Frozen registration</a> · <a href="preflight_receipt.json">Candidate preflight</a> · <a href="review_bundle/implementation.patch">Implementation patch</a> · <a href="test_receipt.json">Focused tests</a>')
    html='<!doctype html><html><head><meta charset="utf-8"><title>OLR targeted repairs v4</title><style>body{font:16px/1.6 system-ui;background:#eef2f5;color:#172b40;margin:0}main{max-width:1250px;margin:30px auto;padding:38px;background:white}table{border-collapse:collapse;font-size:12px;width:100%;margin:20px 0}td,th{border-bottom:1px solid #dce4ed;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf0f8}img{width:100%;height:auto}h2{margin-top:38px}a{color:#2363a3}</style></head><body><main>'+body+'</main></body></html>'
    (t.OUT/'report.html').write_text(html,encoding='utf-8')
    class Plain(HTMLParser):
        def __init__(self):super().__init__();self.parts=[];self.skip=False
        def handle_starttag(self,tag,attrs):
            if tag=='style':self.skip=True
            if tag in ('p','h1','h2','tr'):self.parts.append('\n\n')
        def handle_endtag(self,tag):
            if tag=='style':self.skip=False
            if tag in ('td','th'):self.parts.append(' | ')
        def handle_data(self,value):
            if not self.skip:self.parts.append(value)
    parser=Plain();parser.feed(html);(t.OUT/'report.txt').write_text(''.join(parser.parts).strip()+'\n',encoding='utf-8')
    names=('report.html','report.txt','results.png','results.json','selected_research_policy.json','qualified_alternatives.json','mechanism_diagnostics.json','rescue_signal_review.json','registration.json','test_receipt.json','preflight_receipt.json','review_bundle/implementation.patch')
    n.write(t.OUT/'completion.json',dict(utc=s.utc(),files={name:n.sha(t.OUT/name) for name in names},promotion=False,independent_validation=False))
    print({'report':str(t.OUT/'report.html'),'selected':selected['label']})

if __name__=='__main__':main()
