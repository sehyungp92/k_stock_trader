"""Readable research decision, native evidence, and a frozen implementation bundle."""
from __future__ import annotations
import html
from html.parser import HTMLParser
from pathlib import Path
import shutil
import ast
import sqlite3
from collections import Counter
import numpy as np
from scripts import olr_targeted_risk_study_v1 as t
n,s,b=t.n,t.s,t.b
LABELS={'guard_control':'Previous 1R guard','auction_priority':'Auction priority only',
    'priority_aggressive_budget':'Aggressive joint budget','priority_balanced_budget':'Tighter joint budget',
    'aggressive_budget_six_slots':'Aggressive budget + sixth slot',
    'priority_position_budget':'Position budget only','priority_sector_account_budget':'Sector/account budgets only'}


def peak_day_attribution(selected):
    """Exact mark-to-exit accounting where the peak-day cohort fully closes."""
    folder=Path(selected['windows']['oos']['folder'])
    daily=n.read(folder/'daily.json');trades=n.read(folder/'trades.json')
    peak=max(n.read(folder/'equity.json'),key=lambda x:x[1]);day=peak[0][:10]
    index=next(i for i,row in enumerate(daily) if row['day']==day)
    if not index or peak[0][11:16]!='09:00' or daily[index]['positions']:return None
    previous,close=daily[index-1],daily[index]
    positions={ast.literal_eval(k)[1]:v for k,v in previous['positions'].items()}
    cohort={symbol:next((x for x in trades if x['symbol']==symbol and x['entry_fill_time'][:10]==previous['day']),None)
        for symbol in positions}
    if any(v is None or v['qty']!=positions[sym] or v['exit_fill_time'][:10]!=day for sym,v in cohort.items()):return None
    study=n.read(t.ROOT/'runtime'/selected['id']/'oos/baseline/study.json')
    database=Path(study['code'])/study['database']
    mark=previous['cash'];rows=[]
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as conn:
        for symbol,qty in positions.items():
            price=conn.execute('select c from bars where symbol=? and ts=?',(symbol,peak[0].replace(' ','T'))).fetchone()[0]
            trade=cohort[symbol]
            sell_fee=trade['commission']-qty*trade['entry_price']*.0002
            rows.append({'symbol':symbol,'qty':qty,'peak_mark_price':price,'average_exit_price':trade['exit_price'],
                'net_change_peak_to_close':qty*(trade['exit_price']-price)-sell_fee,
                'final_trade_net_pnl':trade['net_pnl'],'exit_reason':trade['exit_reason']})
            mark+=qty*price
    error1=abs(mark-peak[1]);error2=abs(sum(v['net_change_peak_to_close'] for v in rows)-(close['equity']-peak[1]))
    assert error1<1e-5 and error2<1e-5
    result={'candidate_id':selected['id'],'peak_time':peak[0],'peak_nav':peak[1],
        'peak_day_close_nav':close['equity'],'contributions':rows,
        'peak_nav_reconstruction_error':error1,'pnl_reconciliation_error':error2,
        'source_completion_sha256':n.sha(folder/'completion.json'),'database_hash':study['database_hash'],
        'interpretation':'Exact marked-value-to-actual-exit accounting, not a claim that the 09:00 bar close was executable. Previous-day cash and full held quantities reconcile the native peak; every position is closed that day.'}
    n.write(t.OUT/'selected_peak_day_attribution.json',result)
    return result


def table(headers, rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def plot(data,selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    from zoneinfo import ZoneInfo
    kst=ZoneInfo('Asia/Seoul')
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    control=next(row for row in data['candidates'] if row['label']=='guard_control')
    for j,w in enumerate(('train','oos')):
        for title,folder,color in [('Original saved',data['original_saved'][w]['folder'],'#9ca3af'),
            ('Previous 1R guard',control['windows'][w]['folder'],'#d97706'),
            ('Selected research policy',selected['windows'][w]['folder'],'#047857')]:
            rows=n.read(Path(folder)/'equity.json');values=np.array([v[1] for v in rows])
            dates=[datetime.fromisoformat(v[0]) for v in rows]
            axes[0,j].plot(dates,100*(values/1e8-1),color=color,label=title,lw=1.7)
            dd=100*(values/np.maximum.accumulate(np.r_[1e8,values])[1:]-1)
            axes[1,j].plot(dates,dd,color=color,label=title,lw=1.7)
        for ax in axes[:,j]:
            ax.grid(alpha=.15)
            ax.xaxis.set_major_locator(mdates.MonthLocator(tz=kst))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y',tz=kst))
            ax.set_xlim(dates[0].replace(day=1,hour=0,minute=0,second=0),
                dates[-1].replace(hour=23,minute=59,second=59))
        axes[0,j].set(title=('IS: 7 Aug 2025–31 Mar 2026' if w=='train' else 'Inspected OOS: 1 Apr–10 Jul 2026'),ylabel='Net return (%)')
        axes[1,j].set(ylabel='Intraday drawdown (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('OLR targeted structural experiments — return and drawdown',fontsize=17,fontweight='bold')
    fig.savefig(t.OUT/'results.png',dpi=170,facecolor='white');fig.savefig(t.OUT/'results.svg');plt.close(fig)


def main():
    data=n.read(t.OUT/'results.json');decision=n.read(t.OUT/'selected_research_policy.json')
    selected=next(row for row in data['candidates'] if row['id']==decision['selected_id'])
    control=next(row for row in data['candidates'] if row['label']=='guard_control')
    peak_day=peak_day_attribution(selected)
    plot(data,selected)
    p=lambda v:'<p>'+v+'</p>'
    h=lambda v:'<h2>'+v+'</h2>'
    cash=lambda v:f'₩{v/1e6:+.3f}m'
    num=lambda v:f'{v:.4f}'
    label=LABELS[selected['label']]
    body='<h1>OLR: targeted exit, risk and frequency experiments</h1>'
    body+=p('<strong>Research decision: '+html.escape(label)+'.</strong> Status: '+decision['status']+'. Seven fixed primary variants were tested using native execution, original signal scoring and actual IS-to-OOS market-history carry. New mechanisms remain disabled by default in ordinary strategy configuration; the selected research profile is saved separately.')
    a,z=selected['windows']['train'],selected['windows']['oos']
    body+=p('The selected nominal result is <strong>'+num(a['return_pct'])+'% IS return / '+num(a['drawdown_pct'])+'% max DD / '+str(a['trades'])+
        ' trades</strong>, and <strong>'+num(z['return_pct'])+'% OOS return / '+num(z['drawdown_pct'])+'% max DD / '+str(z['trades'])+
        ' trades</strong>. Relative to original saved round_1, IS return changes by '+f"{a['return_pct']-data['original_saved']['train']['return_pct']:+.4f}"+
        ' percentage points, OOS return by '+f"{z['return_pct']-data['original_saved']['oos']['return_pct']:+.4f}"+
        ' points, and OOS drawdown by '+f"{z['drawdown_pct']-data['original_saved']['oos']['drawdown_pct']:+.4f}"+
        ' points. This is the best qualifying choice under the registered ranking among the tested variants, not proof of a global or future-return optimum.')
    body+=p('IS is 7 August 2025–31 March 2026 (156 sessions). The inspected OOS is 1 April–10 July 2026 (69 sessions). Each scoring account starts flat with ₩100m. Returns include native costs and capacity constraints. Maximum drawdown below uses intraday marked equity, including open positions. April–July has been repeatedly inspected and is development evidence.')
    body+=p('Equity-curve timestamps are source bar labels. A completed five-minute bar becomes usable at its end; an exit condition on a 15:00 bar is not a decision available at 15:00. The corrected auction example executes at the next eligible 15:05 open after the completed interval and cancellation acknowledgement.')
    body+='<img src="results.png" alt="Native equity and intraday drawdown comparisons">'
    body+=h('Results and the return-preservation decision')
    rows=[]
    for row in data['candidates']:
        a,z=row['windows']['train'],row['windows']['oos']
        rows.append([LABELS[row['label']],num(a['return_pct']),num(a['drawdown_pct']),a['trades'],
            num(z['return_pct']),num(z['drawdown_pct']),z['trades'],
            'Reference' if row['label']=='guard_control' else ('Pass' if row.get('nominal_qualified') else ', '.join(k for k,v in row.get('gates',{}).items() if not v))])
    body+=table(['Policy','IS return %','IS max DD %','IS trades','OOS return %','OOS max DD %','OOS trades','Nominal gates'],rows)
    body+=p('For comparison, the original saved round_1 returned '+num(data['original_saved']['train']['return_pct'])+'% IS and '+num(data['original_saved']['oos']['return_pct'])+'% OOS; its OOS intraday drawdown was '+num(data['original_saved']['oos']['drawdown_pct'])+'%. The previous continuously warmed 1R policy is the immediate control. Both comparators matter: improving on an already weakened control alone would not establish preservation of original IS performance.')
    body+=p('Before outcomes, acceptance required IS return within 3 percentage points of both the previous guard and original saved baseline, IS drawdown no more than 2 points above the guard, OOS return above the guard, and an OOS drawdown reduction of at least 2 points. The 3-point tolerance is an operational research threshold, not a tolerance expressly supplied by the user. Selection favors the OOS return/drawdown ratio among passing profiles, with IS return and trade count as tie-breakers. Return/risk, IS-preservation and balanced-frequency leaders receive combined stress checks when distinct. The frequency coverage clarification was recorded before its outcomes and before any stress results; gates and primary ranking were unchanged. A mechanical-only fallback is explicitly distinguished if the full drawdown target is not met.')
    body+=p('The <a href="qualified_alternatives.json">qualified alternatives</a> preserve the distinct stress-checked choices rather than implying that return, drawdown and frequency have one universal optimum. A policy can be attractive for frequency while another has the higher return/drawdown ratio. Untested combinations of the component ablations are not qualified by this study.')
    body+=h('What changed in execution')
    body+=p('Previously, any pending exit prevented the completed-bar managed-exit checks from running. This included a less-aggressive closing-auction limit order. The new optional priority rule evaluates the existing managed-exit conditions while an identified next-close auction is pending. If a condition fires, it records a persistent cancellation request and submits no second sell. Only the matching CANCELLED acknowledgement releases a market replacement, sized to the remaining reconciled position. Partial fills while cancellation is pending reduce that replacement; a complete fill removes it. Duplicate acknowledgements cannot create another exit. Already pending market liquidations and non-triggered auctions remain in force.')
    priority=next(row for row in data['candidates'] if row['label']=='auction_priority')
    for w in ('train','oos'):
        comp=priority['trade_comparison'][w]
        changed=[v for v in comp['all_matches'] if v['old_exit']!=v['new_exit'] or abs(v['old_unit_pnl']-v['new_unit_pnl'])>1e-6]
        body+=p(('IS' if w=='train' else 'OOS')+' priority-only accounting: '+str(len(changed))+' matched trades changed per-share outcomes or exit time; total P&amp;L change '+cash(comp['pnl_delta'])+', comprising '+cash(comp['per_unit_execution_effect'])+' per-unit execution, '+cash(comp['quantity_effect'])+' quantity effects and '+cash(comp['new_minus_removed_effect'])+' new/removed opportunities.')
    case=[v for v in priority['trade_comparison']['oos']['all_matches'] if v['symbol']=='042700' and v['entry_decision_time'].startswith('2026-05-14')]
    if case:
        v=case[0]
        body+=p('The diagnosed 042700 auction case now has native evidence: quantity '+str(v['old_qty'])+' → '+str(v['new_qty'])+', exit '+v['old_exit']+' → '+v['new_exit']+', net P&amp;L '+cash(v['old_pnl'])+' → '+cash(v['new_pnl'])+'. '+('The entry quantity and entry price are identical, so this case isolates the exit intervention within the native account replay.' if v['old_qty']==v['new_qty'] and v['old_entry_price']==v['new_entry_price'] else 'Its account quantity or entry execution changed; it is not an isolated fixed-quantity exit comparison.'))
    body+=h('How the risk budgets work')
    body+=p('The entry loss scenario per share is <code>1.25 × original R + estimated entry price × round-trip cost rate</code>. R is the strategy’s native ATR-based risk unit with its existing floors. The cost rate includes two commissions, sell tax and two slippage allowances: 34bp under the baseline assumptions, 54bp under combined stress. The estimated entry price is the existing buffered sizing price. The 1.25 multiplier is a predeclared heuristic allowance for adverse execution; it is not an empirically guaranteed gap or halt quantile.')
    body+=table(['Variant','Position scenario loss / NAV','Same-sector loss / NAV','Total loss / NAV','Slots'],[
        ['Aggressive joint','4%','6%','8%',5],['Tighter joint','3%','4.5%','6%',5],
        ['Position ablation','4%','Off','Off',5],['Sector/account ablation','Off','6%','8%',5],
        ['Frequency extension','4%','6%','8%',6]])
    body+=p('Native ranks, signal thresholds, cash funding and liquidity limits are retained. Risk clipping occurs jointly with funding so unused cash can reach lower qualifying ranks. Existing positions plus the unfilled portion of pending buys consume the loss envelope. Pending sales do not release it. The redistribution pass also includes the cohort’s already approved quantities, so top-ups cannot bypass the caps. The existing liquidity-only recycling pool is kept separate from capital freed by a risk cut.')
    increments=[]
    for row in data['candidates']:
        if 'budget_increment_vs_priority' not in row:continue
        a,z=(row['budget_increment_vs_priority'][w] for w in ('train','oos'))
        increments.append([LABELS[row['label']],num(a['return_gain_pp']),num(a['dd_change_pp']),a['trade_change'],
            num(z['return_gain_pp']),num(z['dd_change_pp']),z['trade_change']])
    body+=table(['Increment versus auction-priority only','IS return pp','IS DD pp','IS trades','OOS return pp','OOS DD pp','OOS trades'],increments)
    body+=p('These component comparisons separate the exit repair from sizing. The joint budget can reduce correlated exposures that a per-position limit admits, while a per-position limit can reduce a high-ATR single name that remains below the sector or account allowance. Cash reallocation and subsequent compounding interact, so joint results are not the sum of the two ablations.')
    body+=p('These are entry-commitment constraints. They do not force liquidation merely because falling NAV later breaches a percentage, and they do not cap realized loss in a gap, halt or missing-liquidity interval. The execution model still requires a prior contiguous completed interval for continuous-open capacity. No intrabar stop fill, opening-auction liquidity or halt-time fill was invented. The first adverse five-minute interval can still account for most of a loss; changing a completed-bar threshold cannot recover information that was unavailable before that interval closed.')
    overshoots=selected['windows']['oos']['realized_position_scenario_overshoots']
    if overshoots:
        body+=table(['Observed scenario overshoot','Entry','Budget % of decision NAV','Realized loss % of decision NAV'],
            [[v['symbol'],v['entry'],num(v['position_scenario_budget_pct']),num(v['realized_loss_pct_decision_nav'])] for v in overshoots])
        body+=p('The observed overshoot is a direct reminder that the sizing allowance is not an execution-price guarantee. The total account scenario budget also cannot prevent cumulative losses across successive trades from producing a larger peak-to-trough drawdown.')
    body+=h('Trading frequency and preservation of profitable opportunities')
    rows=[]
    for row in data['candidates']:
        a,z=row['windows']['train'],row['windows']['oos']
        rows.append([LABELS[row['label']],a['entry_days'],f"{a['trades_per_21_sessions']:.2f}",
            z['entry_days'],f"{z['trades_per_21_sessions']:.2f}",f"{z['win_rate']*100:.2f}",f"{z['profit_factor']:.3f}"])
    body+=table(['Policy','IS entry days','IS trades / 21 sessions','OOS entry days','OOS trades / 21 sessions','OOS win rate %','OOS profit factor'],rows)
    body+=p('Counts refer to native aggregated trade lifecycles, not execution fragments. The sixth-slot test adds one native-ranked opportunity without loosening the entry thresholds. More trades are useful only if they pass the return and drawdown gates; a frequency increase alone is not evidence of more alpha.')
    six=next(row for row in data['candidates'] if row['label']=='aggressive_budget_six_slots')
    for w in ('train','oos'):
        v=six['sixth_slot_increment'][w];c=v['decomposition']
        body+=p('Sixth slot versus the identical five-slot budget, '+w+': trades '+f"{v['trade_change']:+d}"+', return '+f"{v['return_gain_pp']:+.4f}"+'pp, DD '+f"{v['dd_change_pp']:+.4f}"+'pp. New-minus-removed trade P&amp;L is '+cash(c['new_minus_removed_effect'])+'; matched-trade quantity effects are '+cash(c['quantity_effect'])+'. This separates extra opportunities from dilution/reallocation of the first five ranks.')
        if c['added']:
            biggest=max(c['added'],key=lambda x:x['net_pnl'])
            body+=p('There are '+str(c['new_count'])+' added and '+str(c['removed_count'])+' removed lifecycles in that comparison. The largest added trade is '+biggest['symbol']+' on '+biggest['entry_decision_time'][:10]+', contributing '+cash(biggest['net_pnl'])+'. This concentration matters when interpreting the apparent benefit of the extra slot.')
    inventory=n.read(t.OUT/'opportunity_inventory.json')
    body+=p('The unchanged snapshots explain the frequency headroom: there are '+str(len(inventory['train']['sixth_candidate_dates']))+' IS dates with at least six candidates, but only '+str(len(inventory['oos']['sixth_candidate_dates']))+' OOS date. OOS has 32 empty candidate snapshots out of 67, versus 54 out of 154 IS. Each scoring window also retains its existing two terminal runoff dates without new entry snapshots. Market-history continuity does not manufacture candidates on otherwise empty signal dates. A sixth slot therefore has limited OOS opportunity coverage; substantially more OOS frequency would require evidence for a broader signal-admission rule, which this exit/sizing experiment does not claim to establish.')
    rows=[]
    for w in ('train','oos'):
        c=selected['trade_comparison'][w]
        rows.append([w,c['matched'],c['new_count'],c['removed_count'],c['winner_to_loser'],
            f"{c['matched_winner_pnl_retention_pct']:.2f}%",f"{c['top_five_original_winner_pnl_retention_pct']:.2f}%",
            cash(c['per_unit_execution_effect']),cash(c['quantity_effect']),cash(c['new_minus_removed_effect'])])
    body+=table(['Window','Matched','Added','Removed','Winner→loser','Matched winner P&L retained','Top-five winner P&L retained','Per-unit effect','Quantity effect','New−removed effect'],rows)
    body+=p('The decomposition reconciles exactly to the account P&amp;L difference. Per-unit execution effects include entry as well as exit fill changes, because a quantity change can alter participation across bars. Winner labels are used only for retrospective diagnosis; no eventual MAE, MFE or P&amp;L enters the policy. Quantity reductions can reduce profitable-trade participation even when no winner becomes a loser, which is why winner P&amp;L retention and top-five participation are reported explicitly.')
    critical=[]
    for v in selected['trade_comparison']['oos']['all_matches']:
        if (v['symbol'],v['entry_decision_time'][:10]) in {
            ('042700','2026-05-14'),('042660','2026-05-27'),('329180','2026-05-27'),
            ('055550','2026-06-05'),('403870','2026-06-15')}:
            critical.append([v['symbol'],v['entry_decision_time'][:10],v['old_qty'],v['new_qty'],
                cash(v['old_pnl']),cash(v['new_pnl']),cash(v['pnl_delta'])])
    body+=table(['Previously diagnosed loss','Entry date','Guard qty','Selected qty','Guard P&L','Selected P&L','Change'],critical)
    comparison=selected['trade_comparison']['oos']
    improvements=sorted(comparison['all_matches'],key=lambda v:v['pnl_delta'],reverse=True)[:3]
    body+=p('The three largest matched-trade improvements sum to '+cash(sum(v['pnl_delta'] for v in improvements))+
        ', against total OOS improvement '+cash(comparison['pnl_delta'])+'. This is accounting concentration, not an executable counterfactual that excludes those trades. The small number of tail events limits the evidence for future expected returns, even when the changes are causally implementable.')
    body+=h('Stress, compounding and remaining drawdown')
    original_a,original_z=(data['original_stress'][w] for w in ('train','oos'))
    rows=[['Original saved baseline',num(original_a['return_pct']),num(original_a['drawdown_pct']),original_a['trades'],
        num(original_z['return_pct']),num(original_z['drawdown_pct']),original_z['trades'],'Reference']]
    for row in data['candidates']:
        if len(row['stress'])!=2:continue
        a,z=row['stress']['train'],row['stress']['oos']
        rows.append([LABELS[row['label']],num(a['return_pct']),num(a['drawdown_pct']),a['trades'],
            num(z['return_pct']),num(z['drawdown_pct']),z['trades'],
            'Reference' if row['label']=='guard_control' else ('Pass' if row.get('stress_qualified') else ', '.join(k for k,v in row.get('stress_gates',{}).items() if not v))])
    body+=table(['Combined stress','IS return %','IS DD %','IS trades','OOS return %','OOS DD %','OOS trades','Pass'],rows)
    body+=p('Combined stress reduces modeled participation from 1% to 0.5% and increases slippage from 5bp to 15bp; commissions and sell tax remain 2bp per side and 20bp. The candidate carries its own matching stressed IS history into OOS. Acceptance retains the original saved stressed IS preservation gate, not merely the newer guard comparator. A passed native audit validates the declared historical model and accounting; phase-specific live liquidity remains unverified.')
    qualified=n.read(t.OUT/'qualified_alternatives.json')['policies']
    other=[v for v in qualified if v['candidate_id']!=selected['id']]
    for v in other:
        a,z=v['nominal']['train'],v['nominal']['oos']
        body+=p('The qualified alternative '+LABELS[v['label']]+' preserves more nominal IS return at '+num(a['return_pct'])+
            '%, with OOS return '+num(z['return_pct'])+'% and DD '+num(z['drawdown_pct'])+'%. It is retained as a separate research profile; the selected policy ranks higher under the registered OOS return/drawdown criterion.')
    position=next(row for row in data['candidates'] if row['label']=='priority_position_budget')
    if position.get('stress') and not position.get('stress_qualified'):
        body+=p('The position-only variant increases nominal OOS trades to '+str(position['windows']['oos']['trades'])+
            ', but is not a qualified recommendation: its stressed IS return '+num(position['stress']['train']['return_pct'])+
            '% is below the original-baseline preservation threshold '+num(data['original_stress']['train']['return_pct']-3)+
            '%. The threshold was not relaxed to obtain an OOS frequency improvement.')
    for w in ('train','oos'):
        a=control['capped'][w];z=selected['capped'][w]
        body+=p(('IS' if w=='train' else 'OOS')+' without upward compounding: control return '+num(a['return_pct'])+'%, selected '+num(z['return_pct'])+'%; intraday DD '+num(a['drawdown_pct'])+'% → '+num(z['drawdown_pct'])+'%; trades '+str(a['trades'])+' → '+str(z['trades'])+'. This retains native cash reuse and execution feedback and is not a fixed-share exit experiment.')
    a=selected['windows']['oos'];ep=a['intraday_drawdown']
    body+=p('Selected OOS maximum drawdown: '+num(ep['drawdown_pct'])+'%, from '+ep['peak_time']+' to '+ep['trough_time']+', with NAV '+cash(ep['peak_nav'])+' → '+cash(ep['trough_nav'])+'. Daily-close maximum drawdown is '+num(a['daily_close_drawdown']['drawdown_pct'])+'%. Recovery after the trough within the window: '+str(ep['recovered_after_trough'])+'. Reporting only closed trades or daily marks would conceal part of the exposure path.')
    if peak_day:
        rows=peak_day['contributions']
        winners=sum(v['net_change_peak_to_close'] for v in rows if v['final_trade_net_pnl']>0)
        body+=p('On the peak day itself, NAV declines from '+cash(peak_day['peak_nav'])+' to '+cash(peak_day['peak_day_close_nav'])+
            ' at the close. Holdings that still finish as profitable trades contribute '+cash(winners)+
            ' of that retreat. A lower drawdown therefore requires controlling some unrealized-profit retreat as well as loss trades. This does not establish that a tighter profit exit would preserve IS; earlier blanket tightening failed that test. The values below reconcile previous-day cash, native quantities, retained 09:00-bar closing marks and actual exits to within ₩0.00001. The marked peak is not assumed to be an available executable sale price.')
        body+=table(['Peak-day holding','Shares','Peak bar mark','Average actual exit','Peak-to-exit change, net','Final trade net P&L'],
            [[v['symbol'],v['qty'],f"{v['peak_mark_price']:,.2f}",f"{v['average_exit_price']:,.2f}",
                cash(v['net_change_peak_to_close']),cash(v['final_trade_net_pnl'])] for v in rows])
    body+=h('Chronology and uncertainty')
    rows=[]
    for w in ('train','oos'):
        for month,values in selected['windows'][w]['months'].items():
            rows.append([w,month,values['sessions'],num(control['windows'][w]['months'][month]['return_pct']),
                num(values['return_pct']),num(selected['monthly_comparison'][w][month])])
    body+=table(['Window','Month','Sessions','Guard return %','Selected return %','Gain pp'],rows)
    for w in ('train','oos'):
        proof=selected['bootstrap'][w];ci=proof['return_gain_pp_percentiles']
        body+=p(('IS' if w=='train' else 'OOS')+' paired five-session block resampling (4,000 replications): return-gain percentiles 2.5/50/97.5 = '+', '.join(num(ci[k]) for k in ('2.5','50','97.5'))+' percentage points; fraction positive '+f"{100*proof['fraction_positive']:.1f}%"+'. This is conditional development-sample sensitivity, not a selection-adjusted confidence statement about future returns.')
    body+=p('The monthly table is a chronology diagnostic, not an untouched walk-forward selection exercise. These seven trials follow extensive earlier research on the same windows. Repeatedly inspecting a holdout changes its evidential role; limited local improvements cannot establish the globally optimal rule or expected future return. See the primary research on <a href="https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf">backtest overfitting</a>. No later source-qualified evaluation was performed here; the retained intraday archive used for this study ends on 10 July.')
    body+=h('History continuity, implementation and reproducibility')
    body+=p('Scoring remains the existing formula, including its legacy current-sector volume treatment. The earlier corrected-panel experiment materially weakened IS, so this study does not silently activate that separate model change. Continuous lagged feature reconstruction and actual terminal liquidity-state carry are preserved; they are distinct from changing the current-sector scoring formula.')
    proof=selected['windows']['oos']['authentication']
    body+=p('The August IS start remains an inherited scoring-calendar boundary from the 60-session training plus one-session embargo schedule, not evidence that older signal history is unavailable. Pre-scoring observations remain in the feature reconstruction. The OOS account restores the selected candidate’s actual IS terminal market-history state, ending 31 March, before the first 1 April decision. '+str(proof['verified_prior_observations'])+' retained observations were checked against OOS retained OHLCV. Candidate parameters, execution scenario and compounding mode must match; an independently reset account does not reset its market-history inputs. Both disabled-feature controls reproduce the prior guard’s economic trades, daily balances and intraday marks exactly.')
    coverage={};rows=[]
    for policy in (control,selected):
        values={}
        for w in ('train','oos'):
            folder=Path(policy['windows'][w]['folder']);trades=n.read(folder/'trades.json')
            sparse=[v for v in trades if v['route_metadata']['capacity_forecast']['exit']['status']=='insufficient_history']
            first=min(trades,key=lambda v:v['entry_decision_time'])
            values[w]={'completion_sha256':n.sha(folder/'completion.json'),'insufficient_history_trades':len(sparse),
                'insufficient_history_pnl':sum(v['net_pnl'] for v in sparse),
                'sample_counts':dict(Counter(len(v['route_metadata']['capacity_forecast']['exit']['samples']) for v in sparse)),
                'first_trade':{'symbol':first['symbol'],'decision_time':first['entry_decision_time'],
                    'exit_forecast':first['route_metadata']['capacity_forecast']['exit']}}
            rows.append([LABELS[policy['label']],w,str(len(sparse))+' / '+str(len(trades)),cash(values[w]['insufficient_history_pnl'])])
        coverage[policy['id']]=values
    n.write(t.OUT/'history_coverage.json',coverage)
    body+=table(['Inherited sparse auction coverage','Window','Trades with fewer than 3 samples','Their observed net P&L'],rows)
    first=coverage[selected['id']]['oos']['first_trade'];forecast=first['exit_forecast']
    body+=p('The first selected OOS trade is '+first['symbol']+' at '+first['decision_time']+', with '+str(len(forecast['samples']))+
        ' retained auction sample(s), forecast status '+forecast['status']+'. '+
        ('Its earliest retained sample is '+forecast['samples'][0][0]+'. ' if forecast['samples'] else '')+
        'Literal continuity does not imply dense history for every symbol: the inherited feed observes only the instruments it receives, and the estimator requires at least three eligible observations. Sparse support is a coverage/model limitation separate from resetting at the IS/OOS boundary. The descriptive P&amp;L split is not a causal test of banning sparse forecasts; doing so would also remove a large profitable IS cohort. This study preserves that observation contract rather than claiming that all older market data were ingested into every liquidity estimate.')
    prior=next(c for c in n.read(t.PARENT/'candidate_plan.json')['candidates'] if c['label']=='literal_carry_original')
    prior_folder=t.PARENT/'cases'/prior['id']/'oos/100000000_baseline'
    s.forensic.authenticate(prior_folder);prior_result=n.read(prior_folder/'result.json')
    body+=p('The earlier matched boundary repair alone changed original OOS return from '+num(data['original_saved']['oos']['return_pct'])+'% to '+num(prior_result['return_pct'])+
        '% and intraday DD from '+num(data['original_saved']['oos']['drawdown_pct'])+'% to '+num(prior_result['drawdown_pct'])+
        '%. It repaired continuity but did not improve that observed return; its small economic effect does not explain the original large drawdown. The present gains come from the separately tested exit and allocation changes.')
    body+=p('49 distinct focused checks passed, including delayed and duplicate cancellation acknowledgements, partial/full-fill races, serialization, pending-buy reservations, top-up risk limits, allocation permutation invariance, default behavior and native next-open sequencing. Every reported native window passed the existing full execution/ledger audit and ended without inventory or pending orders. Default source behavior remains unchanged economically; feature flags are explicit in the research profile.')
    body+=p('Review files: <a href="selected_research_policy.json">selected full research profile</a>, <a href="results.json">authenticated results and trade-level decomposition</a>, <a href="review_bundle/implementation.patch">implementation patch</a>, <a href="implementation_receipt.json">source identity receipt</a>, <a href="test_receipt.json">test receipt</a>. Raw native replay folders and completion hashes are recorded in the results. The next evidential step is to freeze this profile for a genuinely later, unseen period or forward shadow execution, with the same history and execution contract. Further changes should respond to new evidence rather than additional tuning of these inspected dates.')
    doc='<!doctype html><html><head><meta charset="utf-8"><title>OLR targeted structural experiments</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#172033;background:#f8fafc;max-width:1450px;margin:32px auto;padding:0 30px 60px}h1{font-size:32px;line-height:1.2}h2{margin-top:36px;font-size:23px}p{max-width:1200px}img{width:100%;background:white;border:1px solid #dbe1ea}.table{overflow:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:10px;text-align:right;border:1px solid #dbe1ea;vertical-align:top}th:first-child,td:first-child{text-align:left}th{background:#e7eef5}a{color:#0369a1}code{background:#e7eef5;padding:2px 4px}</style></head><body>'+body+'</body></html>'
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
    plain=Text();plain.feed(doc);(t.OUT/'report.txt').write_text(''.join(plain.parts),encoding='utf-8')
    for name in ('olr_targeted_risk_study_v1.py','olr_targeted_risk_results_v1.py','olr_targeted_risk_qualify_v1.py','olr_targeted_risk_report_v1.py','olr_targeted_risk_verify_v1.py'):
        shutil.copy2(t.REPO/'scripts'/name,t.OUT/'review_bundle'/name)
    tests=t.OUT/'review_bundle/tests/strategy_olr'
    tests.mkdir(parents=True,exist_ok=True)
    shutil.copy2(t.REPO/'tests/strategy_olr/test_priority_loss_budget.py',tests/'test_priority_loss_budget.py')
    files={str(p.relative_to(t.OUT)).replace('\\','/'):n.sha(p) for p in t.OUT.rglob('*') if p.is_file() and p.name not in ('completion.json','delivery_checks.json')}
    n.write(t.OUT/'completion.json',{'completed_utc':s.utc(),'files':files,'selected':decision['selected_id'],
        'primary_candidates':7,'promotion':False,'independent_validation':False})
    print({'report':str(t.OUT/'report.html'),'files':len(files)},flush=True)


if __name__=='__main__':main()
