"""Report the bounded experiment, including a valid no-upgrade conclusion."""
from __future__ import annotations
from collections import Counter
import difflib
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import olr_opening_risk_study_v2 as t
from scripts import olr_opening_risk_prepare_v2 as preparation
n,s,b=t.n,t.s,t.b
LABELS={'selected_control':'Current selected control','opening_tail_es90':'Lagged opening-loss sizing',
    'concentration_55':'55% single-name cap','weak_partial_half':'One-time half exit on weak follow-through',
    'qualified_pair_union':'Qualified pair combination'}


def p(text):return '<p>'+text+'</p>'
def h(text):return '<h2>'+escape(text)+'</h2>'
def table(head,rows):
    return '<div class="table"><table><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in head)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def num(v):return f'{v:,.3f}'
def cash(v):return f'{v/1e6:+,.3f}m KRW'


def bundle():
    root=t.OUT/'review_bundle';root.mkdir(parents=True,exist_ok=True);patches=[];records=[]
    for w in ('train','oos'):
        parent=n.read(t.PARENT/'runtime'/t.REFERENCE/w/'baseline/study.json')
        for name in preparation.PATCHED:
            before=Path(parent['code'])/name;after=t.ENGINES/w/'inputs/files'/name
            dest=root/w/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(after,dest)
            patches.extend(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True) if before.exists() else [],
                after.read_text(encoding='utf-8').splitlines(True),fromfile='a/'+w+'/'+name,tofile='b/'+w+'/'+name))
            records.append({'window':w,'name':name,'before':n.sha(before) if before.exists() else None,'after':n.sha(after)})
    (root/'implementation.patch').write_text(''.join(patches),encoding='utf-8')
    for src in sorted((t.REPO/'scripts').glob('olr_opening*_v2.py')):shutil.copy2(src,root/src.name)
    test=t.REPO/'tests/strategy_olr/test_opening_risk_v2.py';shutil.copy2(test,root/test.name)
    n.write(t.OUT/'implementation_receipt.json',{'changed_sources':records,'default_off':True,'promotion':False})


def chart(data):
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex='col')
    colors=['#334155','#2563eb','#d97706','#059669','#7c3aed']
    for i,row in enumerate(data['candidates']):
        for j,w in enumerate(('train','oos')):
            equity=n.read(Path(row['windows'][w]['folder'])/'equity.json')
            dates=np.array([np.datetime64(x[0][:19]) for x in equity]);values=np.array([x[1] for x in equity])/1e8
            dd=(values/np.maximum.accumulate(np.r_[1.,values])[1:]-1)*100
            axes[0,j].plot(dates,(values-1)*100,label=LABELS[row['label']],color=colors[i],lw=1.4)
            axes[1,j].plot(dates,dd,color=colors[i],lw=1.3)
    for j,title in enumerate(('IS: 7 Aug 2025 – 31 Mar 2026','Inspected OOS: 1 Apr – 10 Jul 2026')):
        axes[0,j].set_title(title);axes[0,j].set_ylabel('Net cumulative return (%)')
        axes[1,j].set_ylabel('Intraday drawdown (%)');axes[1,j].tick_params(axis='x',rotation=25)
    for ax in axes.flat:ax.grid(alpha=.2);ax.axhline(0,color='#94a3b8',lw=.6)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=10)
    fig.suptitle('OLR: targeted opening risk, concentration and weak-trade exits',fontsize=17)
    fig.tight_layout(rect=(0,.12,1,.95))
    for suffix in ('png','svg'):fig.savefig(t.OUT/('results.'+suffix),dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    t.activate();data=n.read(t.OUT/'results.json');profile=n.read(t.OUT/'selected_research_policy.json')
    assert n.sha(t.OUT/'results.json')==profile['results_sha256']
    assert (t.ROOT/'qualification_complete.json').exists()
    control=data['candidates'][0];selected=next(r for r in data['candidates'] if r['id']==profile['selected_id'])
    chart(data);bundle();refs=data['references']
    adopted=profile['status']=='qualified_research_successor'
    body='<h1>OLR targeted opening-risk experiments</h1>'
    body+=p('<strong>'+('Qualified research successor: '+escape(LABELS[selected['label']]) if adopted else
        'Retain the current selected research profile: no tested successor passed every registered gate.')+'</strong>')
    body+=p('This study tested three specific mechanisms inferred from the previous drawdown diagnosis. Every primary variant was replayed natively in both periods. Only individually qualifying mechanisms could be combined, and only nominally qualifying leaders could become successors after combined execution stress. This is a bounded comparison, not proof of a global optimum or higher future expected returns. April–July has been inspected repeatedly and is development evidence. No live policy was activated.')
    body+='<img src="results.png" alt="Net return and intraday drawdown for all tested variants in both periods">'
    rows=[]
    comparisons=[('Original saved',refs['original']),('Prior loss-exit guard',refs['guard']['windows'])]+[(LABELS[r['label']],r['windows']) for r in data['candidates']]
    for label,windows in comparisons:
        rows.append([label,*[value for w in ('train','oos') for value in (num(windows[w]['return_pct']),num(windows[w]['drawdown_pct']),windows[w]['trades'])]])
    body+=table(['Nominal policy','IS return %','IS max DD %','IS trades','OOS return %','OOS max DD %','OOS trades'],rows)
    body+=p('All returns are net replay returns on separate 100m KRW starting accounts. Maximum drawdown uses every native intraday equity mark, including unrealized profit retreat. Trade counts collapse partial exit legs into unique entry lifecycles; extra sell orders do not count as extra trading opportunities. IS contains 156 sessions and OOS 69 sessions, including terminal runoff. Percentages are not annualized.')
    body+=h('What was changed and why')
    body+=p('The prior diagnosis reconciled the selected OOS drawdown of 16.281% to the native ledger. Four losing trades explained 66.79% of its peak-to-trough decline, while three eventual winners explained another 27.29% through profit retreat. The shipbuilding cohort reached about 75.28% of NAV and a financial holding about 64.05%. Two large losers, 042660 and 403870, had only mild entry-day weakness and were already beyond −1R on their first next-session completed bars; both exited at the first eligible 09:05 open. Their combined losses represented about 42.5% of the drawdown decline. A tighter completed-bar trigger cannot manufacture an earlier executable fill for those gaps.')
    body+=p('1. <strong>Lagged opening-loss sizing.</strong> The scenario per share becomes the larger of the existing 1.25 ATR allowance and estimated adverse close-to-next-eligible-open loss, plus round-trip costs. The estimate uses a 60-session panel, the worst 10% mean, and symbol estimates shrunk toward same-sector peers with a prior strength of 20 observations. Fewer than 20 peer observations invoke the declared-universe peer pool. Own-symbol observations are excluded from its prior pool. Position, sector and account allowances remain 4%, 6% and 8% of decision NAV. This tests exposure before a gap; it neither changes signal ranks nor promises a realized-loss cap.')
    body+=p('2. <strong>55% single-name allocation cap.</strong> This reduces the existing 65% native allocation ceiling while retaining six slots, signal eligibility, ranks and liquidity-only recycling. It limits isolated concentration without applying a blanket reduction to every trade. It does not automatically create new eligible signals or guarantee that released cash will be invested.')
    body+=p('3. <strong>One-time partial weak-trade exit.</strong> On a later session, after three distinct completed bars of that session, a position closing at or below −0.4R with observed MFE below +0.5R requests sale of half its remaining shares, rounded down and leaving at least one share. Only one attempt is allowed per lifecycle. The original stronger full exit has precedence. A pending partial can be replaced only after its matching cancellation acknowledgment; subsequent full size uses actual remaining shares. Rejection or expiry cannot generate repeated partial orders. This targets gradual failure while retaining some recovery exposure; it cannot identify eventual losers with certainty.')
    body+=p('The generic tighter profit trail was excluded: earlier isolated-price diagnostics damaged significant winners and did not support an IS-preserving improvement. Those diagnostics were not full portfolio-policy tests. The present native results, which include cash reuse, partial fills, costs, execution capacity and compounding, determine the adoption decision.')
    body+=h('Selection gates and failures')
    body+=p('Nominal IS return must remain within 3 percentage points of both the original and guard, and within 1.5 points of the current selected control. IS DD may be at most 2 points above the guard. OOS return must improve by at least 0.25 point and OOS DD decline by at least 0.75 point versus the selected control, with at least 180 IS and 45 OOS trades. Under combined stress, all IS return and DD preservation gates still apply; OOS return must be at least the stressed selected control and DD strictly lower. These are explicit acceptance constraints, not fitted estimates of future certainty.')
    rows=[]
    for row in data['candidates'][1:]:
        rows.append([LABELS[row['label']],str(row.get('nominal_qualified',False)),', '.join(k for k,v in row['gates'].items() if not v) or 'None',
            str(row.get('stress_qualified')) if row['stress'] else 'Not advanced',
            ', '.join(k for k,v in row.get('stress_gates',{}).items() if not v) or ('None' if row['stress'] else 'Not advanced')])
    body+=table(['Variant','Nominal passed','Nominal failed gates','Stress passed','Stress failed gates'],rows)
    combo=n.read(t.ROOT/'combination_selection.json');leaders=n.read(t.ROOT/'stress_selection.json')
    body+=p(('One combination was registered from the two individually qualified leaders.' if combo['parents'] else
        'No combination was authorized by the registered rule because fewer than two mechanisms qualified individually.')+
        ' The stressed alternative count is '+str(len(leaders['finalists']))+'. No thresholds were changed after viewing results. Among fully qualified variants the ranking favors OOS return divided by DD, then balanced frequency, then IS return.')
    body+=h('Execution stress and compounding')
    rows=[]
    for label,windows in [('Original saved',refs['original_stress']),('Prior guard',refs['guard']['stress'])]+[(LABELS[r['label']],r['stress']) for r in data['candidates'] if r['stress']]:
        rows.append([label,*[v for w in ('train','oos') for v in (num(windows[w]['return_pct']),num(windows[w]['drawdown_pct']),windows[w]['trades'])]])
    body+=table(['Combined stress','IS return %','IS max DD %','IS trades','OOS return %','OOS max DD %','OOS trades'],rows)
    body+=p('Combined stress uses 0.5% bar participation and 15bp slippage, versus nominal 1% participation and 5bp slippage. Native commissions, sell taxes, phase eligibility, halts, auctions, pending-order reservations and ledger audits remain in force. This is one adverse execution scenario, not a bound on every possible market disruption.')
    rows=[]
    for row in data['candidates']:
        if row['capped']:
            rows.append([LABELS[row['label']],*[v for w in ('train','oos') for v in (num(row['capped'][w]['return_pct']),num(row['capped'][w]['drawdown_pct']),row['capped'][w]['trades'])]])
    body+=table(['No upward compounding','IS return %','IS max DD %','IS trades','OOS return %','OOS max DD %','OOS trades'],rows)
    body+=p('The no-upward-compounding comparison caps allocation equity after gains while retaining native cash reuse and execution feedback. It is a separate exposure comparison, not a fixed-share exit counterfactual or an unregistered replacement selection rule.')
    capped=selected['capped'];nominal=selected['windows']
    body+=p('For the retained policy, capping upward compounding changes IS return by '+num(capped['train']['return_pct']-nominal['train']['return_pct'])+
        ' percentage points, OOS return by '+num(capped['oos']['return_pct']-nominal['oos']['return_pct'])+
        ' points, and reduces OOS DD by '+num(nominal['oos']['drawdown_pct']-capped['oos']['drawdown_pct'])+
        ' points. The OOS improvement comes with a material IS sacrifice, so this comparison does not establish an alternative satisfying the preservation objective.')
    body+=h('Winner preservation, losses and genuine frequency')
    rows=[]
    for row in data['candidates'][1:]:
        for w in ('train','oos'):
            m=row['trade_comparison'][w]
            rows.append([LABELS[row['label']],w,m['matched_winners'],m['winner_to_loser'],m['winner_exit_changes'],
                num(m['matched_winner_pnl_retention_pct']),num(m['top_five_original_winner_pnl_retention_pct']),m['new_count'],m['removed_count'],cash(m['pnl_delta'])])
    body+=table(['Variant','Window','Matched control winners','Turned nonpositive','Winner exit changes','Matched winner P&L retained %','Top 5 winner P&L retained %','Added entries','Removed entries','Net P&L change'],rows)
    body+=p('Winner labels in this table refer to the completed control trades and are diagnostic only; the implemented rules never use these labels. P&L retention includes quantity and compounding effects. An unchanged exit rule can still reduce winning P&L by sizing a winner smaller. A profitable aggregate result is insufficient to claim no winning trade was harmed. The results file also separates per-unit execution changes, quantity changes and added/removed opportunities exactly.')
    weak=next(row for row in data['candidates'] if row['label']=='weak_partial_half')
    rows=[]
    for w in ('train','oos'):
        m=weak['trade_comparison'][w]
        winners=sum(x['pnl_delta'] for x in m['all_matches'] if x['old_pnl']>0)
        losers=sum(x['pnl_delta'] for x in m['all_matches'] if x['old_pnl']<=0)
        rows.append([w,cash(m['pnl_delta']),cash(m['per_unit_execution_effect']),cash(m['quantity_effect']),
            cash(m['new_minus_removed_effect']),cash(winners),cash(losers)])
    body+=table(['Partial-exit effect','Total P&L change','Per-unit execution component','Quantity component','Added minus removed component','Matched control winners','Matched control losers'],rows)
    body+=p('The per-unit, quantity, and added-minus-removed components sum exactly to the total change. The last two columns separately partition matched-trade changes by the control outcome. This is an accounting decomposition, not independent causal experiments: earlier sales change available cash, future allocations and compounding. Increased entry frequency can therefore enlarge later losing exposures or reduce unrelated winners. Per-unit changes include both entry and exit execution; the collapsed final exit reason can remain next_close_exit even when an earlier partial sale changed the weighted exit price.')
    m=weak['trade_comparison']['oos']
    harmed=sorted((x for x in m['all_matches'] if x['old_pnl']>0 and x['new_pnl']<=0),key=lambda x:x['pnl_delta'])
    if harmed:
        body+=p('The OOS control winners made nonpositive by this policy are '+', '.join(
            escape(x['symbol']+' entered '+x['entry_decision_time'][:10])+' ('+cash(x['old_pnl'])+' → '+cash(x['new_pnl'])+')' for x in harmed)+
            '. Their observed temporary weakness satisfied the rule but subsequently recovered in the control. Retaining half the position preserved some subsequent recovery exposure, yet did not preserve these wins. This is direct evidence against adopting this particular weak-follow-through threshold under the stated objective; it does not prove every earlier-exit design would fail.')
    benefits=sorted(m['all_matches'],key=lambda x:x['pnl_delta'],reverse=True)[:3]
    body+=p('The largest OOS paired improvements were '+', '.join(
        escape(x['symbol']+' entered '+x['entry_decision_time'][:10])+' '+cash(x['pnl_delta']) for x in benefits)+
        '. These benefits must be assessed together with the harmed recovery trades and changed sizing, rather than selecting only the successful loss exits.')
    rows=[]
    for row in data['candidates']:
        for w in ('train','oos'):
            q=row['windows'][w];rows.append([LABELS[row['label']],w,q['trades'],q['entry_days'],num(q['trades_per_21_sessions']),num(100*q['win_rate'])])
    body+=table(['Variant','Window','Entry lifecycles','Active entry days','Trades per 21 sessions','Win rate %'],rows)
    body+=p('The inherited selected universe has 32 empty OOS candidate snapshots out of 67 entry-eligible dates, with only one date offering a sixth candidate. Six slots previously added nine IS trades and no OOS trades. Risk and exit improvements can free capacity, but cannot by themselves overcome an empty opportunity set. Signal thresholds were not relaxed merely to increase counts.')
    body+=h('Opening-history evidence and continuous IS/OOS state')
    panel_rows=[];diagnostics={}
    for w in ('train','oos'):
        panel=n.read(t.ENGINES/w/'inputs/files/research/opening_risk_v2/observations.json')
        panel_rows.append([w,len(panel['universe']),len(panel['sessions']),len(panel['observations']),str(panel['exclusions'])])
    body+=table(['Panel','Declared symbols','Retained source sessions','Usable loss observations','Excluded/missing observations'],panel_rows)
    for row in data['candidates']:
        if not row['mutations']['capacity_opening_tail']:continue
        for w in ('train','oos'):
            trades=n.read(Path(row['windows'][w]['folder'])/'trades.json');tail=[]
            for trade in trades:
                meta=trade['route_metadata'];ev=meta['opening_tail_evidence']
                tail.append({'symbol':trade['symbol'],'entry':trade['entry_decision_time'],'through':ev['through'],
                  'tail_fraction':ev['fraction'],'own_observations':ev['symbol_observations'],
                  'allocation_binding_limit':meta['allocation_binding_limit'],
                  'approved_scenario_loss_pct_nav':100*meta['scenario_loss_budget']['approved_loss']/meta['scenario_loss_budget']['equity'],
                  'exceeds_atr_floor':float(meta['estimated_entry_price'])*ev['fraction']>float(meta['risk_per_share'])*1.25})
            diagnostics[row['id']+'_'+w]=tail
            body+=p(escape(LABELS[row['label']])+' '+w+': '+str(sum(x['exceeds_atr_floor'] for x in tail))+' of '+str(len(tail))+
                ' admitted entry lifecycles have a tail scenario above the 1.25 ATR floor. This counts admitted trades, not rejected intents. A largely nonbinding estimate is evidence of limited added protection under these data, not proof that overnight losses cannot exceed the existing allowance.')
            above=[x for x in tail if x['exceeds_atr_floor']]
            limits=Counter(x['allocation_binding_limit'] for x in above)
            identical=n.read(Path(row['windows'][w]['folder'])/'equity.json')==n.read(Path(control['windows'][w]['folder'])/'equity.json')
            body+=p('For the admitted trades whose historical scenario exceeds the ATR floor, the recorded allocation limits are '+
                escape(str(dict(limits)))+'. '+('Every intraday equity mark is exactly identical to the selected control. The added scenario therefore supplies no incremental economic protection in this window: the existing order-size and funding constraints still determine the executed exposure.' if identical else
                'The resulting equity path differs from the selected control; its full return and drawdown effects are reported above.'))
    n.write(t.OUT/'tail_diagnostics.json',diagnostics)
    body+=p('Every observation joins the immediately preceding declared trading day’s official closing match to the first continuous open with a contiguous completed bar supporting at least one share at both registered participation rates. Declared corporate-action ex-dates are excluded. The panel contains all 103 declared names and never selects names by subsequent trading or P&L. Current-session outcomes are ingested only after the session; sizing uses completed prior sessions. Frozen inherited sector labels are not a separately verified point-in-time classification archive, and undeclared corporate actions remain a data limitation.')
    body+=p('Before April 1, each candidate restores its own actual March 31 terminal liquidity state from the matching IS candidate, execution scenario and compounding mode. Tail-enabled runs additionally restore the actual terminal 60-session tail panel and require exact equality with a separately built OOS prefix. All overlapping IS/OOS observation rows are checked for equality. Each account resets flat to 100m KRW; its market history does not reset. The August 7 IS scoring boundary comes from the inherited 60-session training plus one-session embargo schedule. Earlier retained bars remain warm-up inputs; the OOS archive contains 217 prior sessions.')
    coverage=[]
    for row in (control,selected) if selected['id']!=control['id'] else (control,):
        for w in ('train','oos'):
            trades=n.read(Path(row['windows'][w]['folder'])/'trades.json')
            sparse=[x for x in trades if x['route_metadata']['capacity_forecast']['exit']['status']=='insufficient_history']
            coverage.append([LABELS[row['label']],w,len(sparse),len(trades),cash(sum(x['net_pnl'] for x in sparse))])
    body+=table(['Liquidity coverage','Window','Sparse-history trades','All trades','Sparse-cohort net P&L'],coverage)
    body+=p('Dense opening-loss history is separate from the inherited routed auction-liquidity history. The latter remains sparse for some names, including the first selected OOS trade’s single March 10 auction observation. This study does not silently replace that liquidity estimator or claim all historical bars are available to every predictor. Earlier blanket history/scoring changes harmed IS, and banning the sparse cohort would discard substantial profitable IS activity.')
    body+=h('Remaining drawdown and decision')
    q=selected['windows']['oos'];episode=q['intraday_drawdown']
    body+=p('The retained policy’s OOS return is '+num(q['return_pct'])+'%, intraday DD '+num(q['drawdown_pct'])+'%, and trade count '+str(q['trades'])+
        '. Its DD peak is '+escape(episode['peak_time'])+' at '+cash(episode['peak_nav'])+', with trough '+escape(episode['trough_time'])+' at '+cash(episode['trough_nav'])+
        '. Daily-close DD is '+num(q['daily_close_drawdown']['drawdown_pct'])+'%. The bar-mark peak is not assumed to be an executable full-position liquidation price.')
    body+=p(('The successor met every registered nominal and stressed preservation gate. Its profile remains a research candidate pending new-period or forward evidence.' if adopted else
        'No tested alternative achieved the required return, drawdown, frequency and stress combination. The implemented switches remain available for reproducible research, but the selected profile keeps them off. The correct outcome is to retain the current profile, not choose a lower drawdown obtained by an unacceptable sacrifice of IS return or declare an unqualified improvement.')+
        ' Remaining risk includes opening gaps, correlated holdings, profit retreat, halts, sparse liquidation estimates and compounding. No stop rule can guarantee separation of future winners and losers. Further changes should be tested on new source-qualified data or a frozen forward shadow run rather than repeatedly tuning thresholds on these inspected dates.')
    body+=p('Artifacts: <a href="selected_research_policy.json">selected research profile</a>, <a href="results.json">full results and trade decomposition</a>, <a href="qualified_alternatives.json">qualified alternatives</a>, <a href="review_bundle/implementation.patch">implementation patch</a>, <a href="implementation_receipt.json">source identities</a>, <a href="test_receipt.json">focused test evidence</a>, <a href="registration.txt">prior registration</a>. Native completion hashes and audit receipts are embedded in the results.')
    windows=sum(len(row[group]) for row in data['candidates'] for group in ('windows','stress','capped'))
    tests=n.read(t.OUT/'test_receipt.json')['passed']
    body+=p(str(windows)+' native replay windows passed their existing full execution and ledger audits, with no terminal inventory or pending orders. '+str(tests)+
        ' focused tests passed, including next-open partial fills, cancellation acknowledgments, remaining-quantity exits, rejection/expiry handling, history restart continuity and joint loss budgets. Disabled controls reproduce the prior selected profile’s economic trades, daily balances, intraday marks and terminal liquidity history in nominal, stressed and capped comparisons. The separate delivery-check receipt binds the visually reviewed chart, report, source files and final profile to these authenticated results.')
    doc='<!doctype html><html><head><meta charset="utf-8"><title>OLR opening-risk experiments</title><style>body{font:16px/1.65 system-ui;color:#172033;background:#f8fafc;max-width:1450px;margin:30px auto;padding:0 30px 50px}h1{font-size:31px;line-height:1.25}h2{font-size:23px;margin-top:34px}img{width:100%;background:white}.table{overflow:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:9px;border:1px solid #dbe1ea;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}th{background:#e7eef5}a{color:#0369a1}</style></head><body>'+body+'</body></html>'
    (t.OUT/'report.html').write_text(doc,encoding='utf-8')
    class Text(HTMLParser):
        def __init__(self):super().__init__();self.parts=[];self.style=False
        def handle_starttag(self,tag,attrs):
            if tag=='style':self.style=True
            if tag in ('p','h1','h2','tr'):self.parts.append('\n')
        def handle_endtag(self,tag):
            if tag=='style':self.style=False
            if tag in ('td','th'):self.parts.append(' | ')
        def handle_data(self,value):
            if not self.style:self.parts.append(value)
    text=Text();text.feed(doc);(t.OUT/'report.txt').write_text(''.join(text.parts),encoding='utf-8')
    files={p.relative_to(t.OUT).as_posix():n.sha(p) for p in t.OUT.rglob('*') if p.is_file() and p.name not in ('completion.json','delivery_checks.json','background_handoff.txt')}
    n.write(t.OUT/'completion.json',{'selected':profile['selected_id'],'files':files,'promotion':False,'independent_validation':False})
    print({'report':str(t.OUT/'report.html'),'files':len(files)},flush=True)


if __name__=='__main__':main()
