"""Report authenticated precision trials without rewriting result/profile bindings."""
from collections import Counter
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import olr_precision_study_v3 as t
from scripts.olr_opening_risk_report_v2 import p,h,table,num,cash
n,s,b=t.n,t.s,t.b
LABELS={'selected_control':'Retained selected control','liquidation_tail_es90':'Quantity-aware liquidation sizing',
        'structural_partial_half':'Confirmed structural weakness: half exit','late_confirmation_9':'Confirmation through bar 9',
        'risk_plus_late_confirmation':'Registered risk + later-confirmation combination'}


def chart(data):
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    colors=['#334155','#2563eb','#d97706','#059669','#7c3aed']
    for i,row in enumerate(data['candidates']):
        for j,w in enumerate(('train','oos')):
            equity=n.read(Path(row['windows'][w]['folder'])/'equity.json')
            dates=np.array([np.datetime64(x[0][:19]) for x in equity]);values=np.array([x[1] for x in equity])/1e8
            dd=(values/np.maximum.accumulate(np.r_[1.,values])[1:]-1)*100
            axes[0,j].plot(dates,(values-1)*100,label=LABELS[row['label']],color=colors[i],lw=1.3)
            axes[1,j].plot(dates,dd,color=colors[i],lw=1.3)
    for j,title in enumerate(('IS: 7 Aug 2025 to 31 Mar 2026','Inspected OOS: 1 Apr to 10 Jul 2026')):
        axes[0,j].set_title(title);axes[0,j].set_ylabel('Net cumulative return (%)')
        axes[1,j].set_ylabel('Intraday drawdown (%)')
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.axhline(0,color='#94a3b8',lw=.6);ax.tick_params(axis='x',rotation=25)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=10)
    fig.suptitle('OLR: capacity-aware risk, selective exits and additional confirmations',fontsize=16)
    fig.tight_layout(rect=(0,.14,1,.95))
    for suffix in ('png','svg'):fig.savefig(t.OUT/('results.'+suffix),dpi=160,bbox_inches='tight')
    plt.close(fig)


def comparison_table(policies):
    rows=[]
    for label,windows in policies:
        rows.append([label,*[value for w in ('train','oos') for value in
            (num(windows[w]['return_pct']),num(windows[w]['drawdown_pct']),windows[w]['trades'],windows[w].get('entry_days','not recorded'))]])
    return table(['Policy','IS return %','IS DD %','IS trades','IS entry days','OOS return %','OOS DD %','OOS trades','OOS entry days'],rows)


def diagnostics(data):
    output={};control=data['candidates'][0]
    key=lambda r:(r['symbol'],r['entry_decision_time'])
    for row in data['candidates']:
        output[row['id']]={}
        for w in ('train','oos'):
            trades=n.read(Path(row['windows'][w]['folder'])/'trades.json')
            old={key(x):x for x in n.read(Path(control['windows'][w]['folder'])/'trades.json')}
            late=[];tail=[]
            for trade in trades:
                meta=trade['route_metadata'];ev=meta.get('liquidation_tail_evidence')
                if ev:
                    tail.append({'symbol':trade['symbol'],'entry':trade['entry_decision_time'],**ev,
                        'exceeds_atr_floor':meta['estimated_entry_price']*ev['fraction']>1.25*meta['risk_per_share'],
                        'allocation_binding_limit':meta.get('allocation_binding_limit'),
                        'qty_changed_vs_control':key(trade) in old and trade['qty']!=old[key(trade)]['qty'],
                        'qty':trade['qty'],'net_pnl':trade['net_pnl']})
                if meta.get('late_confirmation_evidence',{}).get('late'):
                    late.append({'symbol':trade['symbol'],'entry':trade['entry_decision_time'],
                        'bar_number':meta['late_confirmation_evidence']['bar_number'],'qty':trade['qty'],
                        'entry_notional':trade['qty']*trade['entry_price'],'net_pnl':trade['net_pnl'],
                        'new_vs_control':key(trade) not in old})
            sparse=[x for x in trades if x['route_metadata']['capacity_forecast']['exit']['status']=='insufficient_history']
            output[row['id']][w]={'tail_admitted':tail,'late_admitted':late,
                'sparse_auction_trades':len(sparse),'sparse_auction_pnl':sum(x['net_pnl'] for x in sparse),
                'identical_intraday_path':n.read(Path(row['windows'][w]['folder'])/'equity.json')==n.read(Path(control['windows'][w]['folder'])/'equity.json')}
    return output


def main():
    t.activate();data=n.read(t.OUT/'results.json');profile=n.read(t.OUT/'selected_research_policy.json')
    complete=n.read(t.ROOT/'qualification_complete.json')
    assert profile['results_sha256']==complete['results_sha256']==n.sha(t.OUT/'results.json')
    assert complete['profile_sha256']==n.sha(t.OUT/'selected_research_policy.json')
    control=data['candidates'][0];selected=next(r for r in data['candidates'] if r['id']==profile['selected_id'])
    adopted=profile['status']=='qualified_research_successor';refs=data['references'];diag=diagnostics(data)
    n.write(t.OUT/'mechanism_diagnostics.json',diag);chart(data)
    body='<h1>OLR: precise risk and opportunity experiments</h1>'
    body+=p('<strong>'+('Qualified research successor: '+escape(LABELS[selected['label']]) if adopted else
        'No tested successor met every fixed requirement; retain the selected research policy.')+'</strong>')
    body+=p('The aim is higher returns and genuine entry frequency with lower intraday OOS drawdown, subject to preserving IS performance. The retained selected policy is the starting point. This is a bounded experiment with three primary mechanisms and at most one predefined interaction, not evidence of a global optimum or proven higher future expected returns. April–July has been repeatedly inspected and remains development evidence. No live policy was activated.')
    body+='<img src="results.png" alt="Native cumulative net returns and intraday drawdown in IS and inspected OOS">'
    body+=comparison_table([('Original saved',refs['original']),('Prior loss-exit guard',refs['guard']['windows'])]+[(LABELS[r['label']],r['windows']) for r in data['candidates']])
    body+=p('Returns are net replay returns on separate 100m KRW accounts, not annualized. IS has 156 sessions; OOS has 69, including terminal runoff. DD uses every native intraday equity mark, including unrealized profit retreat. Trade counts collapse partial sells into unique entry lifecycles. Extra sell orders are not additional trading opportunities. Entry days count dates with actual first fills.')
    body+=h('What the completed trials establish')
    bylabel={row['label']:row for row in data['candidates']}
    structural=bylabel['structural_partial_half'];tail=bylabel['liquidation_tail_es90'];late=bylabel['late_confirmation_9']
    sa,sz=structural['windows']['train'],structural['windows']['oos'];ca,cz=control['windows']['train'],control['windows']['oos']
    body+=p('<strong>The selective exit is a useful nominal improvement, but not a fully qualified successor.</strong> IS return rose '+num(sa['return_pct']-ca['return_pct'])+' percentage points and OOS return rose '+num(sz['return_pct']-cz['return_pct'])+' points. IS entries increased from '+str(ca['trades'])+' to '+str(sa['trades'])+', with IS DD increasing '+num(sa['drawdown_pct']-ca['drawdown_pct'])+' points. OOS entries stayed at '+str(sz['trades'])+' and DD fell only '+num(cz['drawdown_pct']-sz['drawdown_pct'])+' points. It passed the return and IS-preservation requirements, but missed both the 47-entry OOS floor and the 0.75-point DD reduction requirement. It was therefore not admitted to successor stress under the fixed protocol; its stress robustness is untested here.')
    sm=structural['trade_comparison']['oos'];before=cz['intraday_drawdown'];after=sz['intraday_drawdown']
    fast=[x for x in sm['all_matches'] if x['symbol'] in ('042660','403870') and x['old_pnl']<0]
    body+=p('Why the return improvement did not produce a comparable DD improvement: the largest paired OOS benefit was 034730 entered May 11, with '+cash(next(x['pnl_delta'] for x in sm['all_matches'] if x['symbol']=='034730' and x['entry_decision_time'].startswith('2026-05-11')))+' saved before the common May 15 drawdown peak. The two dominant fast-opening losers still exited at the same 09:05 prices; their combined P&L change was '+cash(sum(x['pnl_delta'] for x in fast))+' as the changed cash/quantity path slightly increased their exposure. Both policies peaked and troughed at the same timestamps. The peak rose '+cash(after['peak_nav']-before['peak_nav'])+' and the trough rose '+cash(after['trough_nav']-before['trough_nav'])+', leaving only '+cash((before['peak_nav']-before['trough_nav'])-(after['peak_nav']-after['trough_nav']))+' less absolute peak-to-trough decline. Protecting gradual failures is useful, but it leaves the dominant later opening shocks largely intact.')
    tm=tail['trade_comparison']['oos'];fast_tail=[x for x in tm['all_matches'] if x['symbol'] in ('042660','403870') and x['old_pnl']<0]
    winner_cut=next(x for x in tm['all_matches'] if x['symbol']=='034020' and x['entry_decision_time'].startswith('2026-04-23'))
    body+=p('<strong>The new liquidation estimate became economically active, but its selectivity was inadequate.</strong> It improved the combined P&L of the two fast losers by '+cash(sum(x['pnl_delta'] for x in fast_tail))+'. However, 034020 entered April 23 alone lost '+cash(-winner_cut['pnl_delta'])+' of control profit, mostly through reduced quantity. Aggregate matched-winner P&L retention was '+num(tail['trade_comparison']['train']['matched_winner_pnl_retention_pct'])+'% IS and '+num(tm['matched_winner_pnl_retention_pct'])+'% OOS. The scenario reduced exposure to good and bad trades together; its IS return cost was '+num(ca['return_pct']-tail['windows']['train']['return_pct'])+' points. This is evidence against adopting this specific unconditional conservative envelope, not evidence that size and liquidation risk are irrelevant.')
    late_is=diag[late['id']]['train']['late_admitted'];late_oos=diag[late['id']]['oos']['late_admitted']
    body+=p('<strong>More confirmation opportunities did not provide enough additional alpha.</strong> The '+str(len(late_is))+' admitted IS entries confirmed in bars 7–9 produced '+cash(sum(x['net_pnl'] for x in late_is))+' in aggregate. The '+str(len(late_oos))+' OOS late entry, 112610 on April 22, had '+cash(sum(x['entry_notional'] for x in late_oos))+' of real entry notional and '+cash(sum(x['net_pnl'] for x in late_oos))+' net P&L. The IS total return fell '+num(ca['return_pct']-late['windows']['train']['return_pct'])+' points, including changed allocations and displaced opportunities. The frequency increase was real, but economically costly. The combination retained much of the selective-exit OOS benefit while importing the late-entry IS deterioration, so it could not satisfy the joint objective.')
    body+=p('These results separate three distinct problems: gradual loss management, exposure to fast opening shocks, and the economic quality of additional entries. They support preserving the selective-exit implementation as a research alternative, while retaining the selected control under the unchanged acceptance rules. They do not justify relaxing those rules or treating the untested structural variant as stress-qualified.')
    body+=h('Why this round differs from the unsuccessful opening-risk round')
    body+=p('The previous one-share tail rule did not change executed quantities; the 55% concentration cap sacrificed IS return; and the broad weak-follow-through half exit also cut temporary weakness in eventual winners. That round offered no dedicated way to admit an additional valid signal and allowed combinations only after components qualified individually. These failures motivated the present mechanisms and the registered interaction rule; no thresholds were changed after new native outcomes.')
    body+=p('<strong>Correction to the earlier diagnosis:</strong> the old tail panel uses the first capacity-eligible continuous open, generally 09:05, not simply 09:00. It already includes the first five-minute fall for the 042660 and 403870 examples. The new experiment addresses the remaining one-share capacity proxy and previous-close reference; it does not claim to repair a missing first opening bar.')
    body+=p('The selected control previously returned 12.190% OOS with 16.281% intraday DD. Four losing trades explained 66.79% of its peak-to-trough decline and three eventual winners another 27.29% through profit retreat. The 042660 and 403870 losses alone explained about 42.5%; both had crossed the 1R guard before their first eligible 09:05 exits. A tighter completed-bar trigger cannot create earlier executable liquidity. The mechanisms therefore distinguish exposure before a fast loss, persistent later weakness, and unused confirmation opportunities.')
    body+=h('The registered mechanisms')
    body+=p('<strong>Quantity-aware liquidation sizing.</strong> The existing 4/6/8% position/sector/account scenario allowances remain unchanged. Per-share risk is the larger of the original 1.25 ATR and a lagged adverse entry-reference-to-liquidation estimate, plus unchanged costs. The reference is the last completed historical continuous close before the 14:30 decision, adjusted for session hours. This is a decision-time price proxy, not each historical trade’s actual entry fill. Requested current notional is translated to historical shares. Eligible next-session continuous opens consume only a participation-limited amount of the immediately preceding contiguous bar’s volume. Native session, halt and periodic-auction exclusions remain.')
    body+=p('The estimate uses 60 prior sessions, the mean of the worst 10%, and symbol/peer shrinkage with 20-observation prior strength. Same-sector peers exclude the symbol; sparse sector support falls back to the declared universe. The adverse average-price envelope is monotone in requested size. It is quoted at the full proposed notional including permitted recycled extras, so allocation cannot bypass its sizing evidence. Any remainder without supported continuous liquidity is valued at zero in this conservative scenario. That is an impairment assumption, not a forecast that the stock becomes worthless. It may overstate risk, especially when auction liquidity could exist; the IS and frequency gates reject excessive shrinkage. Aligned-date losses are recorded for diagnosis, not represented as a newly estimated portfolio-correlation cap.')
    body+=p('<strong>Selective structural half exit.</strong> The existing later-session rule—at least three completed bars, close at or below −0.4R, and observed MFE below +0.5R—also requires two contiguous completed closes below both the persisted entry-session low and cumulative current-session VWAP. Sell half once, retain the rest, and preserve full-guard precedence, real cancellation acknowledgments, capacity restrictions and rejection/expiry latches. Entry-session lows survive later partial entry fills and restart. These are causal conditions available at the decision; eventual loser labels are never policy inputs. Exploratory winner separation on these same dates is not a guarantee of preserving future winners.')
    body+=p('<strong>Three additional confirmation bars.</strong> All original bars 1–6 and their price, VWAP, close-location, ranking and capacity requirements are preserved. A signal that first qualifies in bars 7–9 may be admitted using real remaining cash, gross exposure allowance, risk allowance and executable capacity. Earlier rejected signals are not retried. This targets an observed timing bottleneck without broadly lowering quality thresholds or manufacturing trades on empty candidate dates.')
    combination=n.read(t.ROOT/'combination_selection.json')
    parent_names=[LABELS[next(row['label'] for row in data['candidates'] if row['id']==cid)] for cid in combination['parents']]
    body+=p('The sole conditional combination joins the late-confirmation mechanism with the economically active risk component selected by the registered order: IS preservation first, then largest nominal OOS DD reduction, then IS return, then OOS return. Components need not pass the combined frequency objective individually. The rule selected '+escape(' + '.join(parent_names) or 'no active risk component')+'. No additional interaction search was allowed.')
    body+=h('Admission and qualification gates')
    body+=p('IS return may fall at most 3 percentage points versus both the original and guard and 1.5 points versus the selected control; IS DD may rise at most 2 points versus the guard. OOS return must improve at least 0.25 points and OOS DD must fall at least 0.75 points. Native entry lifecycles must reach at least 186 IS and 47 OOS, with at least 88 and 29 entry days. These are simultaneous conditions, not alternatives. The OOS requirement therefore demands at least one extra genuine entry over the 46-trade retained control.')
    body+=table(['Candidate','Nominal qualification','Failed nominal requirements','Stress qualification'],[
        [LABELS[r['label']],'control' if r['id']==control['id'] else r['nominal_qualified'],
         ('reference, not an applicant' if r['id']==control['id'] else ', '.join(k for k,v in r['gates'].items() if not v) or 'none'),
         ('control' if r['id']==control['id'] else r.get('stress_qualified','not eligible / not run'))] for r in data['candidates']])
    body+=p('Combined stress preserves all original/guard/selected IS return and guard-DD requirements. OOS return must be at least the stressed control’s and DD strictly lower; IS/OOS entries must each be at least the matched stressed control. Only nominally qualified objective leaders proceed to stress, at most two. Selection among full qualifiers ranks return/DD, balanced genuine frequency and then IS return. A better single metric is insufficient.')
    body+=h('Combined execution stress and compounding')
    stress=[('Original saved stress',refs['original_stress']),('Guard stress',refs['guard']['stress'])]
    stress += [(LABELS[r['label']],r['stress']) for r in data['candidates'] if len(r['stress'])==2]
    body+=comparison_table(stress)
    body+=p('Combined stress halves broker participation from 1% to 0.5% and increases OLR slippage from 5 to 15 basis points; commission remains 2 basis points and modeled sell tax remains 20 basis points. These are inherited research assumptions, not claims about current statutory costs. All native execution, capacity, ledger and terminal-flat checks remain. The frozen replay source, request and manifest bind the settings. Every new variant failed nominal admission, so only the retained control ran combined stress in this round; no new variant is stress-qualified.')
    body+=comparison_table([(LABELS[r['label']]+' — no upward compounding',r['capped']) for r in data['candidates'] if len(r['capped'])==2])
    body+=p('No-upward-compounding runs cap the sizing equity at starting capital when NAV rises while allowing it to fall with losses. They retain the actual cash account and native risk controls. This distinguishes the contribution of reinvesting gains from entry/exit changes; it is not a claim that all capital effects disappear. Each capped OOS run restores the matching capped IS history.')
    body+=h('Winner preservation and exact accounting decomposition')
    rows=[]
    for row in data['candidates'][1:]:
        for w in ('train','oos'):
            m=row['trade_comparison'][w]
            rows.append([LABELS[row['label']],w,m['new_count'],m['removed_count'],m['winner_to_loser'],
                num(m['matched_winner_pnl_retention_pct']),num(m['top_five_original_winner_pnl_retention_pct']),
                cash(m['pnl_delta']),cash(m['per_unit_execution_effect']),cash(m['quantity_effect']),cash(m['new_minus_removed_effect'])])
    body+=table(['Variant','Window','Added entries','Removed entries','Matched wins to nonwins','Matched-winner P&L retained %','Top-five winner P&L retained %','Total P&L delta','Per-unit effect','Quantity effect','Added minus removed'],rows)
    body+=p('The last three components sum exactly to total P&L change. Per-unit effects include entry and exit fills; quantity effects include changed cash and compounding; added/removed trades measure changed opportunities. This is accounting, not independent causal identification. Matched-winner retention excludes removed winners, so the removed counts and top-five retention must also be considered. A partial sale may leave a next_close_exit label on the final collapsed trade; weighted proceeds and economic changes, rather than final reason text alone, establish its effect.')
    body+=h('Mechanism activation and meaningful frequency')
    rows=[];late_rows=[]
    for row in data['candidates']:
        for w in ('train','oos'):
            d=diag[row['id']][w];tail=d['tail_admitted'];late=d['late_admitted'];m=row['windows'][w]
            rows.append([LABELS[row['label']],w,m['trades'],m['entry_days'],len(late),len(tail),
                sum(e['exceeds_atr_floor'] for e in tail),sum(e['qty_changed_vs_control'] for e in tail),d['identical_intraday_path']])
            late_rows.extend([[LABELS[row['label']],w,e['symbol'],e['entry'],e['bar_number'],e['qty'],cash(e['entry_notional']),cash(e['net_pnl'])] for e in late])
    body+=table(['Variant','Window','Unique entries','Entry days','Late entries','Tail-evidenced entries','Tail above ATR','Tail matched quantities changed','Equity exactly control'],rows)
    if late_rows:body+=table(['Variant','Window','Symbol','Decision time','Confirmation bar','Shares','Entry notional','Net P&L'],late_rows)
    body+=p('Tail activation counts above refer to admitted trades; rejected intents are not reconstructed from missing trades. Quantities can differ indirectly because earlier decisions change cash. Additional confirmations must have actual shares, notional and native fills; the table exposes token or negligible additions. The inherited funnel still has 32 empty OOS snapshots among 67 eligible entry dates. Extending confirmation cannot supply signals to those empty dates.')
    body+=h('Continuous history and data limits')
    rows=[]
    for w in ('train','oos'):
        panel=n.read(t.ENGINES/w/'inputs/files/research/precision_v3/liquidation_observations.json')
        rows.append([w,len(panel['universe']),len(panel['sessions']),len(panel['observations']),str(panel['exclusions'])])
    body+=table(['Panel','Declared names','Source sessions','Observations','Excluded or impaired coverage'],rows)
    body+=p('IS begins August 7 because of the inherited 60-session training plus one-session embargo schedule, not because older bars were discarded. The OOS input archive has 217 prior sessions. Every OOS replay restores its own actual March 31 terminal liquidity history for the exact candidate, execution scenario and compounding mode before April 1. Tail-enabled runs also restore their actual IS terminal liquidation panel and match it against an independently constructed OOS prefix. All overlapping panel rows match exactly. The two performance accounts restart flat at 100m KRW; market history remains continuous.')
    coverage=[]
    for row in (control,selected) if selected['id']!=control['id'] else (control,):
        for w in ('train','oos'):
            d=diag[row['id']][w];coverage.append([LABELS[row['label']],w,d['sparse_auction_trades'],row['windows'][w]['trades'],cash(d['sparse_auction_pnl'])])
    body+=table(['Policy','Window','Sparse auction-history trades','All trades','Sparse-cohort P&L'],coverage)
    body+=p('The dense price/liquidation panel does not repair the separately routed auction-volume history. The control has sparse auction support, including only one prior observation for its first OOS entry. The declared 103-name universe and frozen sector taxonomy are not a point-in-time universe/classification archive. Declared corporate-action exclusions do not guarantee coverage of undeclared events. Five-minute opens and volumes approximate the available execution evidence; there is no order-book replay or guaranteed stop price. The conservative liquidation envelope can miss unobserved future gaps and correlation, despite its impairment rule. Previously identified sector-flow feature semantics require their own coherent calibration; this study does not silently change the incumbent scoring bands.')
    body+=h('Decision and remaining risk')
    z=selected['windows']['oos'];episode=z['intraday_drawdown']
    body+=p('Final retained policy: '+escape(LABELS[selected['label']])+'. OOS return '+num(z['return_pct'])+'%, intraday DD '+num(z['drawdown_pct'])+'%, '+str(z['trades'])+' unique entries. Its drawdown peak was '+escape(episode['peak_time'])+' at '+cash(episode['peak_nav'])+' and trough '+escape(episode['trough_time'])+' at '+cash(episode['trough_nav'])+'. Daily-close DD was '+num(z['daily_close_drawdown']['drawdown_pct'])+'%. An intraday marked peak is not assumed to be an executable full-position sale.')
    body+=p(('This successor passed all registered nominal and stress conditions and remains a research candidate.' if adopted else
        'The tests establish why these specific mechanisms were insufficient under the fixed joint objective; they do not establish that improvement is impossible. The starting profile remains selected because none of the tested alternatives earned replacement.')+
        ' High expected returns, more opportunities, winner preservation and smaller drawdowns are competing empirical requirements. No causal rule perfectly identifies future losers. Further claims about expected return need genuinely new source-qualified periods or frozen forward evidence; reranking the inspected April–July sample cannot provide that evidence.')
    body+=p('Artifacts: <a href="selected_research_policy.json">selected research profile</a>, <a href="results.json">full results and matched trades</a>, <a href="qualified_alternatives.json">qualified alternatives</a>, <a href="mechanism_diagnostics.json">activation and late-entry evidence</a>, <a href="review_bundle/implementation.patch">implementation patch</a>, <a href="registration.json">fixed registration</a>, <a href="test_receipt.json">focused test receipt</a>. Native completion and history proofs are embedded in the results.')
    windows=sum(len(r[g]) for r in data['candidates'] for g in ('windows','stress','capped'))
    body+=p(str(windows)+' native windows completed with full execution/capacity/ledger audits and flat terminal inventories. '+str(n.read(t.OUT/'test_receipt.json')['passed'])+' focused tests passed against the frozen engine sources. Control parity covers exact economic trades, daily balances, intraday marks and terminal liquidity history in nominal, stressed and capped runs. Final delivery separately checks the report and chart hashes; visual inspection is required before declaring delivery verified.')
    doc='<!doctype html><html><head><meta charset="utf-8"><title>OLR precision experiments</title><style>body{font:16px/1.65 system-ui;color:#172033;background:#f8fafc;max-width:1520px;margin:30px auto;padding:0 28px 50px}h1{font-size:31px;line-height:1.25}h2{font-size:23px;margin-top:34px}img{width:100%;background:white}.table{overflow:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:9px;border:1px solid #dbe1ea;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}th{background:#e7eef5}a{color:#0369a1}</style></head><body>'+body+'</body></html>'
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
    for src in sorted((t.REPO/'scripts').glob('olr_precision*_v3.py')):shutil.copy2(src,t.OUT/'review_bundle'/src.name)
    for src in (t.REPO/'scripts/olr_liquidation_panel_v3.py',t.REPO/'tests/strategy_olr/test_precision_v3.py'):shutil.copy2(src,t.OUT/'review_bundle'/src.name)
    files={path.relative_to(t.OUT).as_posix():n.sha(path) for path in t.OUT.rglob('*')
           if path.is_file() and path.name not in ('completion.json','delivery_checks.json','background_handoff.txt')}
    n.write(t.OUT/'completion.json',{'selected':profile['selected_id'],'files':files,'promotion':False,'independent_validation':False})
    print({'report':str(t.OUT/'report.html'),'files':len(files)},flush=True)


if __name__=='__main__':main()
