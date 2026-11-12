"""Evidence-led report for residual drawdown; no strategy modifications."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import html
from html.parser import HTMLParser
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo
import numpy as np
from scripts.olr_remaining_drawdown_v2 import ROOT, OUT, PRIOR, REPO, read, write, sha

def table(headers, rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def cash(v): return f'₩{v/1e6:+.3f}m'
def num(v): return f'{v:.2f}'
def p(v): return '<p>'+v+'</p>'
def h(v): return '<h2>'+v+'</h2>'

def plot(static, mechanisms, screens):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax=plt.subplots(2,2,figsize=(15,11),layout='constrained')
    eq=read(ROOT/'oos/equity.json'); dates=[datetime.fromisoformat(v[0]) for v in eq]; nav=np.array([v[1]/1e6 for v in eq])
    kst=ZoneInfo('Asia/Seoul'); a=ax[0,0]
    a.plot(dates,nav,color='#0f766e',lw=1.7); a.fill_between(dates,nav,np.maximum.accumulate(nav),color='#dc2626',alpha=.12)
    ep=static['oos']['intraday_episode']
    a.scatter([dates[ep['peak_index']],dates[ep['trough_index']]],[ep['peak_nav']/1e6,ep['trough_nav']/1e6],color='#b91c1c')
    a.set(title='Selected OOS: 16.28% intraday drawdown',ylabel='Marked equity (KRW million)')
    a.xaxis.set_major_locator(mdates.MonthLocator(tz=kst));a.xaxis.set_major_formatter(mdates.DateFormatter('%b',tz=kst));a.grid(alpha=.15)
    a=ax[0,1]; comps=mechanisms['attribution']; labels=['Shipbuilding pair','403870 loss','055550 loss','Three winners: giveback','042700 residual','010950 temporary dip','Other positions, net']
    vals=[v['drawdown_points'] for v in comps];a.barh(labels[::-1],vals[::-1],color=['#0f766e' if x<0 else '#b45309' for x in vals[::-1]])
    for i,x in enumerate(vals[::-1]):a.text(max(x,0)+.06,i,f'{x:+.2f}',va='center',fontsize=10)
    a.set(title='Exact contribution to the common drawdown',xlabel='Percentage points of original peak NAV',xlim=(-1.3,6.5));a.axvline(0,color='#475569',lw=.7)
    a=ax[1,0];names=['Median ATR / entry price','Median position 1R / NAV'];x=np.arange(2)
    for i,w in enumerate(['train','oos']):
        v=static[w]['risk_distributions']['all']; y=[v['risk_pct']['50'],v['risk_account_pct']['50']]
        a.bar(x+(i-.5)*.34,y,.34,label='IS' if i==0 else 'OOS',color=['#94a3b8','#b45309'][i])
        for xx,yy in zip(x+(i-.5)*.34,y):a.text(xx,yy+.08,f'{yy:.2f}%',ha='center')
    a.set_xticks(x,names);a.set(title='Risk exposure increased, not just stop losses',ylabel='Percent',ylim=(0,6));a.legend(frameon=False);a.grid(axis='y',alpha=.15)
    a=ax[1,1]; keys=['close_minus_0p4R','close_minus_0p8R','next_day_weak_no_followthrough','mature_profit_fade_1R'];x=np.arange(4)
    for i,w in enumerate(['train','oos']):
        vals=[screens[w]['screens'][k]['earlier_hits']['winners'] for k in keys]
        a.bar(x+(i-.5)*.34,vals,.34,label='IS winners' if i==0 else 'OOS winners',color=['#94a3b8','#b45309'][i])
        for xx,yy in zip(x+(i-.5)*.34,vals):a.text(xx,yy+.2,str(yy),ha='center')
    a.set_xticks(x,['−0.4R close','−0.8R close','Weak follow-through','Mature-profit fade'],rotation=12)
    a.set(title='Winners touched before the existing exit condition',ylabel='Observed winning trades');a.legend(frameon=False);a.grid(axis='y',alpha=.15)
    fig.suptitle('OLR remaining drawdown — audited path attribution and diagnostic screens',fontsize=17,fontweight='bold')
    fig.savefig(OUT/'diagnosis.png',dpi=160,facecolor='white');fig.savefig(OUT/'diagnosis.svg');plt.close(fig)

def main():
    static=read(OUT/'static.json'); mechanisms=read(OUT/'mechanisms.json'); screens=read(OUT/'paths_and_screens.json')
    priced=read(OUT/'isolated_price_screens.json')
    diag={w:read(OUT/(w+'_drawdown.json')) for w in ('train','oos')}
    priorout=REPO/'data/backtests/analysis/olr_targeted_risk_20261010'
    prior=read(priorout/'results.json')
    plot(static,mechanisms,screens)
    body='<h1>OLR: why the selected strategy still draws down 16.28%</h1>'
    body+=p('<strong>The residual problem is a combination of concentrated overnight/first-bar risk, slow recognition of some gradual failures, and profit giveback. A blanket tighter stop is not an adequate remedy.</strong> This report diagnoses the selected aggressive six-slot research profile, cc9a4ad98fc93351e, after the previous auction-priority and risk-budget improvements. It does not diagnose the superseded 20.37% drawdown guard as if it were still the selected policy.')
    body+=p('Two new observational native replays, one IS and one OOS, reproduce every saved trade, daily balance, intraday equity mark and terminal market-history state exactly, and pass the original full execution audits. No strategy rule, frozen input, parameter or live configuration was changed. The new work reconstructs position-level cash and marks and screens possible interventions using only information available at each observed decision time. Screens are diagnostics, not new backtest results.')
    body+='<img src="diagnosis.png" alt="Selected OOS drawdown, exact attribution, risk shift and affected winners">'
    body+=h('1. Current performance and the precise drawdown being explained')
    body+=table(['Policy','IS return','IS intraday DD','IS trades','OOS return','OOS intraday DD','OOS trades'],[
        ['Original round_1','55.05%','7.20%',176,'−3.33%','29.16%',47],
        ['Previous 1R guard','54.54%','8.17%',177,'7.43%','20.37%',47],
        ['Selected six-slot risk profile','53.64%','8.18%',186,'12.19%','16.28%',46]])
    ep=diag['oos']['episode']
    body+=p('IS covers 7 August 2025–31 March 2026; OOS covers 1 April–10 July 2026. The selected OOS equity peaks at <strong>₩131.797m on the 15 May 09:00 bar</strong>, falls to <strong>₩110.340m on the 9 July 12:15 bar</strong>, and loses <strong>₩21.458m / 16.2807%</strong>. It has not recovered that peak by 10 July. Ending equity is ₩112.190m, still 14.88% below the peak. Bar labels precede availability: the completed 09:00 five-minute interval is available at 09:05.')
    body+=p('Daily-close maximum DD is 10.94%, but the 16.28% intraday measure is economically relevant and includes open positions. The difference is not a reporting mistake and should not be hidden by changing the metric. On 15 May alone, marked equity falls from ₩131.797m to ₩125.180m by the close; ₩5.855m of that retreat belongs to trades that remain profitable.')
    body+=table(['Source of decline','Change in NAV','DD contribution','Share of net decline'],[[v['component'],cash(v['nav_change']),f"{v['drawdown_points']:+.4f} pp",f"{v['share_decline_pct']:.2f}%"] for v in mechanisms['attribution']])
    body+=p('These components reconcile to the same peak and trough; they are not a sum of all losing trades. The 034730 loss entered on 11 May, for example, is large and relevant to exit design but precedes this peak. The temporary 010950 loss at the trough later recovers to a final profit of ₩0.092m. Cutting every position retrospectively labelled adverse would therefore cut recoveries as well.')
    body+=p('Four post-peak losses—042660, 329180, 403870 and 055550—explain 66.79% of the net decline. The three profitable peak holdings explain another 27.29%. Other gains offset part of the decline. At an unchanged original peak, reducing this episode to 10% would require ₩'+num(mechanisms['unchanged_peak_trough_improvement_required']['10']/1e6)+'m more equity at the old trough. That is scale arithmetic, not a forecast: any executable intervention can change both the peak and trough.')
    body+=h('2. The larger problem is risk carried per opportunity')
    stats=[]
    for label,fn in [('Win rate',lambda x:f"{100*x['summary']['winners']/x['summary']['trades']:.2f}%"),
        ('Average winner',lambda x:cash(x['summary']['winner_pnl']/x['summary']['winners'])),
        ('Average loser',lambda x:cash(x['summary']['loser_pnl']/x['summary']['losers'])),
        ('Profit factor',lambda x:num(x['summary']['winner_pnl']/-x['summary']['loser_pnl'])),
        ('Median ATR / entry price',lambda x:num(x['risk_distributions']['all']['risk_pct']['50'])+'%'),
        ('Median full-position 1R / pre-fill NAV',lambda x:num(x['risk_distributions']['all']['risk_account_pct']['50'])+'%'),
        ('Mean losing trade P&L in original R',lambda x:num(statistics.mean(r['net_pnl']/(r['qty']*r['risk']) for r in x['rows'] if r['net_pnl']<0))+'R')]:
        stats.append([label,fn(static['train']),fn(static['oos'])])
    body+=table(['Observed statistic','IS','OOS'],stats)
    body+=p('The OOS win rate does not collapse: it rises from 47.85% to 50%. The mean losing trade, measured in its own original ATR risk, is also almost unchanged at −0.593R versus −0.587R. Yet the average currency loss grows 2.89 times. This is strong descriptive evidence that larger and differently concentrated exposures, alongside clustered tail events, are central. It does not establish identical conditional return distributions or prove stable future alpha.')
    body+=p('Mean original ATR dollars as a share of decision NAV on losing positions rise from 0.666% to 1.491%; mean net loss per losing position rises from 0.373% to 1.000% of decision NAV. Mean position notional across all trades rises from 20.75% to 28.36% of decision NAV. Median full-position 1R uses total filled quantity divided by the mark preceding its first fill, so it is an exposure diagnostic rather than an exact simultaneous-account allocation measure.')
    body+=p('There are fewer opportunities over which to spread risk: 186 trades on 88 IS entry days versus 46 on 29 OOS entry days. Trades per active entry day fall from 2.11 to 1.59; 18 of 29 OOS entry days have one new traded name, versus 38 of 88 IS days. At the 27 May close, the two shipbuilders together still represent 75.28% of NAV, even after the new sector risk budget. On 5 June, 055550 alone represents 64.05%. Maximum actual gross exposure is 99.42% of NAV, so this is concentrated cash-funded exposure, not evidence of hidden gross leverage.')
    body+=p('The current admission estimate is <code>quantity × [1.25 × original ATR risk + buffered entry price × round-trip costs]</code>, limited to 4% of NAV per position, 6% per sector and 8% across the account. It is neither a realized-loss cap nor a drawdown cap. It does not dynamically sell a deteriorating holding, forecast a particular overnight crash, or prevent a new loss cluster after the previous cohort exits. Successive losses across May and June can therefore accumulate beyond an 8% concurrent allowance.')
    body+=p('The 042660 scenario allows ₩5.003m but realizes a ₩5.292m loss: 4.23% of decision NAV despite a 4% position allowance. The 034730 May loss also exceeds its scenario estimate. For 403870, the scenario allowance is not exceeded, yet its ₩3.830m loss alone costs 3.30% of decision NAV. A scenario can be correctly implemented and still permit a loss that is too large for the desired account drawdown.')
    body+=h('3. Which losing trades can an earlier exit actually help?')
    body+=table(['Position','Entry-day change','True overnight gap','First morning interval','First close → actual exit','Fees','Final P&L'],[
        [v['symbol']+' / '+v['entry_day']]+[cash(v['bridge'][k]) for k in ['entry_to_entry_day_close','overnight_gap','first_bar','first_close_to_actual_exit','fees']]+[cash(v['net_pnl'])] for v in mechanisms['price_bridges']])
    body+=p('The bridge uses all original shares and actual fills and fees. The entry-day and overnight components are separated deliberately; entry-to-next-open is not a pure overnight gap. The first-close-to-exit component includes price movement and can be positive. It must not be labelled wholly avoidable execution delay.')
    body+=p('<strong>042660 and 403870: the first completed bar is already too late for a closer completed-bar threshold to help.</strong> Their first morning closes are −1.326R and −1.049R, and both positions finish selling at the first eligible 09:05 open. On their entry days, the worst observed closes are only −0.171R and −0.032R. A −0.4R, −0.6R or −0.8R completed-bar exit neither catches them the previous day nor advances the morning decision. Their combined final loss is ₩9.122m, 42.51% of the entire peak-to-trough decline. This is a limitation of the current information/execution clock, not proof that all forms of intraday protection are impossible.')
    body+=p('A pre-existing protective order or finer-grained monitoring could address some first-interval decline, but a valid claim needs appropriate higher-frequency price/volume evidence, activation and queue semantics, and gap/limit/halts treatment. It cannot assume execution at the stop price after the market has crossed it. The present profile has <code>hard_stop_enabled=false</code>; its 1R guard submits a market exit after a completed-bar condition. Enabling a nominal hard stop is not a verified solution to the overnight tail.')
    body+=p('<strong>329180: a gradual failure is a better targeted exit candidate.</strong> Its first morning close is only −0.281R, and its highest observed favorable excursion remains below 0.17R. By the 09:10 bar it is approximately −0.477R, but the native 1R guard does not trigger until the 12:25 bar and fills at 12:30. This leaves time to test a failed-follow-through rule that requires weak price action after an initial observation period and absence of genuine follow-through, while retaining the existing exit for successful momentum trades.')
    body+=p('<strong>034730: another gradual failure, outside the maximum-DD interval.</strong> The 11 May entry finishes near flat; the next day it deteriorates through −0.4R around the 09:15 bar, reaches the native condition on the 10:30 bar and sells 89 shares at 10:35 and 22 at 10:40. Its ₩4.593m loss is relevant to total return and recurring tail control. It cannot be added to the attribution of the later 15 May peak.')
    body+=p('<strong>055550: a halt/capacity event requires different reasoning.</strong> The 8 June 09:00 source bar already meets the guard, but the market halt interrupts trading. Native fills occur at 09:40 and 09:45. Prices recover after the first observed close, contributing +₩1.677m before fees to the price bridge; treating that whole delay as damage would be wrong. The official broker notice records a 09:03:42 KOSPI circuit breaker, a 20-minute halt and a subsequent 10-minute call auction. Five-minute source bars overlap those boundaries, so they do not establish finer execution chronology. The existing audit checks the declared conservative participation model; it does not validate otherwise unavailable live fills. <a href="https://www.kakaopaysec.com/customer/notice/dynamicBoardPageDetail.do?id=7275">Official notice</a>.')
    body+=p('A pending-order audit finds four OOS cases with a market liquidation already working and one actionable pending auction, 042700. In that auction case the new priority mechanism cancels and replaces as designed, selling at 15:05 after the 15:00-labelled trigger. The single observation recorded before cancellation is not evidence that the former blocked-exit bug persists. Repeated market-exit conditions while an order works are likewise not grounds to submit duplicate sells.')
    body+=h('4. Preserving winners requires a separate profit-management design')
    body+=table(['Profitable peak holding','Peak → actual exit retreat','Final net profit'],[
        [v['symbol'],cash(v['episode_pnl_delta']),cash(v['final_trade_pnl'])] for v in diag['oos']['contributions'] if v['held_at_peak'] and v['final_trade_pnl']>0])
    body+=p('The current profit rule activates once MFE reaches 1R and exits when completed-bar profit falls below <code>max(0, MFE − 1.25R)</code>. Between 1R and 1.25R MFE this permits a retreat all the way to entry; above that level it still permits 1.25R giveback. Because ATR/price is higher OOS, the same R gap permits a larger percentage-price retreat. Marked high-water profit is not necessarily executable at size, and a stop cannot capture the maximum retrospectively.')
    body+=p('On 15 May, 066570 is +3.830R at the first close, +3.211R on the 09:05 bar, and +2.674R on the 09:10 bar when the current fade triggers. 003550 similarly drops from +6.940R to +5.737R by the 09:10 bar; the existing fade triggers on the 09:15 bar. A narrower trailing distance restricted to already mature profits can act one completed bar earlier in these examples. It is a testable hypothesis, not evidence that its eventual IS profit cost is small. The third winner, 034220, never reaches 2R MFE in this holding and would be deliberately outside that hypothesis.')
    body+=p('Profit preservation has two meanings that must be measured separately: keeping a trade profitable, and retaining its full economic contribution. Converting no winner into a loser is insufficient if a rule repeatedly halves the largest gains. Conversely, selling part of an extended winner can reduce drawdown without eliminating all participation in a continued advance. A small conditional partial exit followed by the existing runner is therefore a more targeted experiment than tightening all positions.')
    body+=h('5. Decision-time overlap: the cost of treating a loser-like path as a loser')
    labels={'close_minus_0p4R':'Close ≤ −0.4R; MFE <4R','close_minus_0p6R':'Close ≤ −0.6R; MFE <4R',
        'close_minus_0p8R':'Close ≤ −0.8R; MFE <4R','entry_day_weak_no_followthrough':'Entry-day weakness after ≥3 held bars',
        'next_day_weak_no_followthrough':'Next-session weakness after ≥3 held bars','mature_profit_fade_1R':'MFE ≥2R; completed-close giveback ≥1R, later session'}
    rows=[]
    for k,label in labels.items():
        a=screens['train']['screens'][k]['earlier_hits'];z=screens['oos']['screens'][k]['earlier_hits']
        rows.append([label,f"{a['losers']} / {a['winners']}",cash(a['winner_pnl']),f"{z['losers']} / {z['winners']}",cash(z['winner_pnl'])])
    body+=table(['Diagnostic rule','IS losers / winners touched earlier','IS touched-winner P&L','OOS losers / winners touched earlier','OOS touched-winner P&L'],rows)
    body+=p('Weakness screens require close ≤−0.4R and MFE <0.5R, with at least three held bars in the current session. Every screen requires an actual eligible completed-bar observation; it excludes positions with an already-working market liquidation and counts only hits strictly before the existing native exit condition. An actionable pending next-close auction remains eligible for the existing cancellation protocol. No eventual P&L or future MFE enters the trigger; eventual winner/loser labels are used only to evaluate overlap afterwards.')
    body+=p('For timing comparison, the native reference is the first actionable managed-risk condition, or the first actual sell order’s submission when no managed-risk condition occurs. Thus these screens intentionally do not exhaust every possible replacement of an already scheduled auction. Three held bars means three observed completed bars of that session, not three consecutive adverse closes; MFE is the position high-water observed so far.')
    body+=p('<strong>These amounts are not predicted policy gains or losses.</strong> Touched-winner P&L is the baseline profit whose path would change, not the amount that would necessarily be lost. Baseline losing P&L is not recoverable benefit. Replacement prices, partial fills, released cash, subsequent sizing, dates of equity highs and the remaining path all require a native replay. Observed-path screens become invalid as full policy simulations once an intervention changes the account. They narrow mechanisms to test; they do not select a new profitable rule.')
    body+=p('Whole-trade low-price MAE illustrates the broader overlap: 22 of 89 IS winners and 5 of 23 OOS winners visit at least −0.4R. At −0.8R the counts are three and one. Those intrabar extrema are a different, hindsight-based diagnostic from the completed-bar screen above. The 010950 position at the actual trough later recovers, and the 034730 winner entered 7 May reaches approximately −0.913R intrabar before finishing profitable. A rule that claims to eliminate only the bad trades must demonstrate how it distinguishes these recoveries using information available then.')
    body+=p('To go beyond hit counts, a second diagnostic prices a hypothetical unchanged-quantity sale at the next contiguous continuous-session open, only where the prior completed bar supplies enough modeled capacity for every share. It includes 5bp slippage and 22bp sell commission/tax, excludes halt dates, incomplete entries and previously partially sold holdings, and never assumes a stop-price fill. It deliberately excludes unsupported trades. It is still not a native policy replay: cancellation latency, future sizing, freed cash, reinvestment and changes in the equity peak are not simulated.')
    rows=[]
    for k in ['close_minus_0p4R','close_minus_0p6R','close_minus_0p8R','next_day_weak_no_followthrough','mature_profit_fade_1R']:
        a,z=priced['train'][k],priced['oos'][k]
        rows.append([labels[k],f"{a['supported_trades']} / {a['excluded_trades']}",cash(a['isolated_total_pnl_change']),
            f"{z['supported_trades']} / {z['excluded_trades']}",cash(z['isolated_total_pnl_change'])])
    body+=table(['Restricted full-sale arithmetic','IS supported / excluded hits','IS P&L change on supported cases','OOS supported / excluded hits','OOS P&L change on supported cases'],rows)
    body+=p('<strong>The apparently attractive exit hypotheses do not yet demonstrate dominance.</strong> At −0.8R, no IS baseline winner is hit earlier, but six of the fifteen price-supported IS losers are made worse by selling before their partial recovery. Net benefit on those IS cases is only ₩0.059m. OOS loses ₩1.148m on five supported cases, largely because an early sale turns the 034730 recovery winner into a loss. This explains why counting avoided bad trades is inadequate.')
    body+=p('The more targeted next-session weakness screen has +₩1.501m isolated OOS benefit across six supported cases, but −₩3.192m across thirty supported IS cases; 29 additional IS and 8 additional OOS hits require multi-bar or otherwise unsupported execution treatment. The OOS bridge includes +₩0.996m on 329180, yet 012450 and 006400 become worse. A partial sale may reduce the cost to recoveries, but that arithmetic does not prove the resulting complete policy passes the preservation gates.')
    body+=p('For the mature-profit 1R fade, all four OOS earlier hits satisfy the one-bar capacity screen. The two 15 May winners improve by approximately ₩1.438m, and an April 066570 winner improves by ₩0.336m; however, exiting 096770 earlier sacrifices ₩1.978m. The net OOS effect is −₩0.205m, while six supported IS cases lose ₩4.122m. The IS damage includes truncating the 086520 and 017670 gains. A blanket mature-profit fade is therefore a poor first candidate even though it visibly improves the worst drawdown day.')
    body+=p('There is a plausible distinction between immediate opening windfalls and gains built later in the session, but the numerical boundary must not be chosen to remove one counterexample. The adverse 096770 example already has +1.990R at its first next-session close. An opening-profit threshold of exactly 2R would conveniently exclude it; that is a warning about threshold fitting, not robust evidence for a free improvement. The ten earlier-hit IS mature-profit cases have first-next-session closes below 2R, leaving little direct IS support for such a narrowly defined event. This distinction needs a broader lagged event sample, modest-threshold stability and independent evidence before it becomes a policy.')
    body+=p('Implementation detail that matters for new rules: the inherited held-bar list can include preceding sessions. These diagnostics explicitly count distinct eligible completed bars on the current date for the three-bar conditions. A future implementation must do the same; trusting the inherited list length as a current-session clock would unintentionally allow first-bar actions. The selected existing one-bar guard is unchanged by this diagnostic correction.')
    body+=h('6. What the previous controlled comparisons already rule out')
    body+=p('The registered tighter joint budget lowers OOS DD to 12.92%, but IS return falls to 41.47%, versus 53.64% for the selected profile and 55.05% originally. That violates the stated preservation objective. The selected profile without upward compounding reaches OOS return 13.18% and DD 13.92%, but IS return drops to 46.67%, a 6.97-point reduction relative to selected. It is valuable evidence about growth in exposure, not an acceptable unconditional replacement under the existing IS gate.')
    body+=p('Earlier authenticated experiments on preceding configurations also warn against simple global tightening: the 0.4R guard costs almost 20 IS return points; a blanket MFE-gap change from 1.25R to 1R produces IS return about 41.04% and OOS return −3.87%. These are not new combinations with the selected budget, so their effects cannot be added to current results. The continuous-history 0.8R guard alone only improves OOS DD from 20.37% to 20.04% while reducing IS return from 54.54% to 52.43%. It is not a demonstrated route to a large residual-DD reduction.')
    body+=p('The selected combined stress still produces IS return 36.27%, OOS return 10.57% and OOS DD 16.84%, versus nominal 53.64%, 12.19% and 16.28%. Stress halves modeled participation to 0.5% and raises slippage from 5bp to 15bp. The residual drawdown does not disappear or radically change under that tested stress, which supports investigating economic risk concentration and price paths. This is not exhaustive stress coverage, particularly for halt/reopening liquidity and tail dependence.')
    body+=h('7. Ranked next steps and a bounded experimental design')
    body+=p('<strong>First priority: model overnight and opening risk before accepting concentrated exposure.</strong> Keep the selected profile as the comparison control. Build a strictly lagged, all-eligible-symbol observation feed for close-to-next-eligible-exit adverse moves and opening liquidity. Candidate-routing-only coverage is not sufficient for a robust tail estimate. Pool and shrink estimates across related names where symbol samples are sparse; record sample counts, age and fallback policy. A candidate sizing scenario can use the larger of the existing ATR allowance and a conservative historical gap-plus-opening adverse-move estimate, plus costs, rather than blindly adding two overlapping price-risk estimates. This is a candidate design, not a calibrated coefficient recommendation.')
    body+=p('Apply the tail scenario jointly to same-sector and strongly co-moving holdings using only lagged data. The 75% shipbuilding concentration and 64% single-bank concentration show why merely adding slots does not ensure diversification. Prefer targeted resizing over a permanent symbol/sector ban, preserve entry eligibility, and redistribute only to other genuinely qualified candidates within the existing funding and aggregate risk rules. Before policy tests, verify a full IS-prefix → OOS replay and boundary restart produce identical forecasts, with no current/future observation entering a historical decision.')
    body+=p('<strong>Second priority: a staged failed-follow-through pilot, with a high burden of proof.</strong> The next-session weakness condition is causally implementable, but its negative IS price screen rules out describing it as an improvement already established. If pursued, make a single small partial-reduction experiment after at least three completed bars of that session, MFE below 0.5R and close ≤−0.4R; retain the existing 1R guard on the residual. A half-position reduction is a simple research choice, not an asserted optimum. Test it alone and abandon it if full native IS preservation fails. Compare all recoveries, including baseline losers that later become less negative. It cannot solve the first-bar crash cases.')
    body+=p('<strong>Third priority: investigate opening-windfall protection; do not deploy the screened generic mature-profit trail.</strong> The generic 2R-MFE/1R-giveback rule has negative isolated economics in both windows, so repeating it as a blanket full exit has low priority. First collect a wider event sample separating rapid opening extension from gradual intraday trends, using only available lagged features. Require stability away from the 1.990R/2R boundary and enough IS observations. Only then predeclare one partial-exit hypothesis, retaining the existing runner. This is an evidence-gathering step, not a claim that the May winners can be harvested without cost.')
    body+=p('<strong>Fourth priority, only if the first three are insufficient: a small portfolio risk reduction after persistent realized deterioration.</strong> Avoid anchoring a throttle to the single retrospectively best 09:00 mark. A causal rule based on completed daily equity, lagged risk and sustained deterioration is more defensible, but can still sell into recoveries and materially reduce IS compounding. Test one mild version separately. Unconditional no-upward compounding and the tighter joint budget already demonstrate the cost of excessive risk reduction.')
    body+=p('Use one control and one predeclared variant per mechanism, serial native runs, and at most one combination of mechanisms that pass individually. Keep all original gates: nominal and stressed IS return within 3 percentage points of both original and guard references, IS DD no more than 2 points worse than the guard, OOS return above guard and at least 2 points of DD improvement versus guard. In addition, require a new successor to improve the selected profile itself; passing an old, weaker control is not progress. Evaluate the return/DD Pareto trade-off against selected without relaxing existing gates after seeing outcomes.')
    body+=p('For each candidate report nominal and combined stress, no-upward-compounding sensitivity, all-native execution/ledger/capacity audits, exact candidate/scenario-specific history carry, trade count and active days, matched-winner and top-five-winner P&L retained, winners converted to losses, previously prevented losses that recover, and per-trade/cluster attribution. Freeze this experiment set and the selection rule first. Given only 46 OOS trades and four dominant post-peak losses, an improvement concentrated in these dates is weak evidence for expected future returns; use a genuinely later unseen period or forward shadow execution before promoting a policy.')
    body+=h('8. History, signal selection and remaining limitations')
    body+=p('The August IS start is the inherited 60-session training plus one-session embargo/scoring-calendar boundary, not the beginning of all market data. The selected OOS restores its actual IS terminal market history through 31 March before 1 April. The prior continuity verification checks 1,156 retained observations against OHLCV; the OOS intraday history spans 217 prior sessions, including pre-IS history and all 156 IS sessions. Each scoring account starts flat with ₩100m, which is distinct from resetting its market-history state. Both new observation runs exactly reproduce the saved terminal state.')
    body+=p('A real limitation remains: preserving routed history does not provide dense history for every symbol. The first OOS trade has only one retained auction-volume sample. In the selected results, 96/186 IS trades with insufficient auction samples earn ₩27.409m while 21/46 OOS trades lose ₩4.765m. A blanket ban would remove productive IS exposure, and the label does not establish causation. Improve causal observation coverage and uncertainty treatment, then test the policy effect separately. The earlier boundary repair alone changed original OOS DD from 29.1622% to 29.1740%; a boundary reset is not the main explanation of the residual losses.')
    body+=p('The low-score cohort earns ₩26.592m IS but loses ₩5.819m OOS, and the highest-volatility cohort (ATR/price ≥6%) remains profitable in both windows. Therefore neither a low-score ban nor a high-volatility ban is justified by these descriptive splits alone. The legacy sector-volume scoring treatment is a separate issue preserved in this study; earlier wholesale correction weakened IS. Do not quietly mix signal changes into an exit/risk comparison.')
    body+=p('Frequency remains constrained by actual signals: 32/67 OOS candidate snapshots are empty, and only one OOS date has at least six candidates. The sixth slot added IS trades but no OOS trades relative to the five-slot joint-budget variant. An exit or sizing improvement can preserve opportunity participation and occasionally free capacity; it cannot manufacture valid signals on empty days. Loosening entry admission is a different hypothesis with its own preservation and validation burden.')
    body+=p('April–July has been extensively inspected and used to choose policies. It is development evidence, not an untouched independent test. These exact reconciliations establish what the declared historical simulator did and when its rules could react; they do not establish that the marked peak was sellable, that historical liquidity generalizes, or that any proposed rule improves future expected return. Zero interference with all eventual winners cannot be promised when winners and losers share early adverse paths.')
    body+=h('9. Reproducibility and evidence')
    body+=table(['Window','Reconstructed intraday marks','Maximum cash/mark reconciliation error','Exact economic parity','Native audit'],[
        [w,diag[w]['marks'],f"₩{diag[w]['all_bar_ledger_reconstruction_max_error']:.10f}",'Trades, daily, equity, terminal history','Passed'] for w in ('train','oos')])
    body+=p('Sources: <a href="static.json">trade/risk cohorts</a>; <a href="oos_drawdown.json">exact OOS ledger attribution</a>; <a href="train_drawdown.json">IS ledger reconstruction</a>; <a href="mechanisms.json">price bridges and exposure</a>; <a href="paths_and_screens.json">decision-time paths and all screen hits</a>; <a href="isolated_price_screens.json">restricted isolated-sale arithmetic</a>; <a href="verification.json">verification and hashes</a>. The full raw observed runs are recorded in those files under <code>'+str(ROOT)+'</code>.')
    body+=p('Unchanged prior artifacts: <a href="../olr_targeted_risk_20261010/report.html">qualified comparison report</a>; <a href="../olr_targeted_risk_20261010/selected_research_policy.json">selected research profile</a>; <a href="../olr_targeted_risk_20261010/review_bundle/implementation.patch">implemented auction-priority and sizing patch</a>; <a href="../olr_drawdown_diagnosis_20261010/report.html">earlier guard diagnosis and exit experiments</a>. This follow-up adds diagnostics and proposed experiments, not a new trading configuration.')
    document='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>OLR remaining drawdown diagnosis</title><style>body{font:17px/1.65 system-ui,sans-serif;color:#18312e;background:#f8faf9;max-width:1180px;margin:35px auto;padding:0 25px 65px}h1{font-size:34px;line-height:1.2}h2{margin-top:42px;font-size:24px;color:#115e59}p{margin:17px 0}strong{color:#124e49}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:10px 12px;border-bottom:1px solid #d7e1de;text-align:left;vertical-align:top}th{background:#e7efec}img{width:100%;height:auto}code{font-size:14px;overflow-wrap:anywhere}a{color:#08776c} @media print{body{font-size:12px;max-width:none}h2{break-after:avoid}.table{overflow:visible}}</style><body>'+body+'</body></html>'
    (OUT/'report.html').write_text(document,encoding='utf-8')
    class Extract(HTMLParser):
        def __init__(self): super().__init__(); self.parts=[];self.suppress=False
        def handle_starttag(self,tag,attrs):
            if tag in ('style','title'):self.suppress=True
        def handle_data(self,data):
            if not self.suppress:self.parts.append(data)
        def handle_endtag(self,tag):
            if tag in ('style','title'):self.suppress=False
            if tag in ('p','h1','h2','tr'):self.parts.append('\n')
            if tag in ('td','th'):self.parts.append(' | ')
    e=Extract();e.feed(document);(OUT/'report.txt').write_text(''.join(e.parts),encoding='utf-8')
    print({'report':str(OUT/'report.html'),'words':len(''.join(e.parts).split())})

def verify(chart_reviewed):
    assert chart_reviewed, 'Visual review of diagnosis.png is required.'
    proofs={}
    for w in ('train','oos'):
        proof=read(ROOT/(w+'_exact_parity.json'));assert proof['status']=='passed'
        for folder in [Path(proof['reference']),ROOT/w]:
            comp=read(folder/'completion.json')
            for name,hsh in comp['artifacts'].items():assert sha(folder/name)==hsh,(folder,name)
            assert read(folder/'execution_audit.json')['audit_pass']
        for name in ('trades.json','daily.json','equity.json','capacity_terminal_state.json'):
            assert read(Path(proof['reference'])/name)==read(ROOT/w/name),(w,name)
        d=read(OUT/(w+'_drawdown.json')); assert d['all_bar_ledger_reconstruction_max_error']<1e-5
        assert abs(sum(v['episode_pnl_delta'] for v in d['contributions'])+d['episode']['decline'])<1e-5
        proofs[w]={'parity':proof,'raw_evidence_hashes':{name:sha(ROOT/w/name) for name in ('managed_state.jsonl.gz','native_journal.jsonl.gz','observer_receipt.json')},
            'marks':d['marks'],'max_reconstruction_error':d['all_bar_ledger_reconstruction_max_error']}
    from PIL import Image
    size=Image.open(OUT/'diagnosis.png').size
    document=(OUT/'report.html').read_text(encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for key,val in attrs:
                if key in ('href','src') and not val.startswith('http'):self.links.append(val)
    parser=Links();parser.feed(document)
    for link in parser.links:
        if link!='verification.json':assert (OUT/link).exists(),link
    profile=REPO/'data/backtests/analysis/olr_targeted_risk_20261010/selected_research_policy.json'
    assert read(profile)['results_sha256']==sha(profile.parent/'results.json')
    write(OUT/'verification.json',{'status':'passed','chart_visually_reviewed':True,'chart_size':size,
        'native_proofs':proofs,'local_links_checked':parser.links,'no_policy_change':True,
        'prior_selected_profile_hash_binding_intact':True,
        'report_and_evidence_hashes':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='verification.json'},
        'analysis_script_hashes':{name:sha(REPO/'scripts'/name) for name in ('olr_remaining_drawdown_v2.py','olr_remaining_drawdown_analysis_v2.py','olr_remaining_drawdown_report_v2.py')}})
    print({'verification':'passed','chart_size':size,'windows':list(proofs)})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');parser.add_argument('--chart-reviewed',action='store_true');args=parser.parse_args()
    if args.verify:verify(args.chart_reviewed)
    else:main()
