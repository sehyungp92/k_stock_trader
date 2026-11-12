"""Render saved qualified evidence without rerunning a results collector."""
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import difflib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts import olr_minute_exit_study_v5 as t
n=t.n
LABELS={'minute_clock_control':'Matched minute control','fixed_minute_loss_and_fade':'Fixed minute loss / MFE exit'}


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def p(text):return '<p>'+escape(text)+'</p>'
def num(value):return '—' if value is None else f'{value:,.3f}'
def money(value):return f'{value/1e6:,.3f}'


def implementation_bundle():
    """Package authenticated research code without changing any frozen input."""
    bundle=t.OUT/'review_bundle';bundle.mkdir(exist_ok=True)
    harness=n.read(t.OUT/'research_harness.json')
    repair=n.read(t.ROOT/'receipt_schema_repair.json')
    assert n.sha(t.OUT/'research_harness.json')==repair['original_harness_sha256']
    sources={}
    for name,digest in harness['files'].items():
        relative=Path(name).as_posix()
        if relative.startswith(('scripts/olr_minute_exit_','tests/strategy_olr/test_minute_')):
            sources['workspace/'+relative]=(t.REPO/name,digest,'frozen qualification harness')
    for name,digest in repair['files'].items():
        source=Path(name)
        if source.suffix!='.py':continue
        if source.is_relative_to(t.REPO):relative='workspace/'+source.relative_to(t.REPO).as_posix()
        else:
            assert source==Path(repair['runtime_qualifier'])
            relative='receipt_schema_repair/qualifier.py'
        sources[relative]=(source,digest,'additive receipt-schema repair')
    required=('snapshots','entry_identity','preflight','resume_controller','clockfix','receipt_resume')
    for part in required:assert 'workspace/scripts/olr_minute_exit_'+part+'_v5.py' in sources,part
    for name in ('olr_minute_exit_report_v5.py','olr_minute_exit_verify_v5.py'):
        source=t.REPO/'scripts'/name
        sources['workspace/scripts/'+name]=(source,n.sha(source),'reporting and delivery verification only')
    patches=[];files={};manifest=[]
    for w in ('train','oos'):
        relative=w+'/implementation.patch';source=bundle/relative
        patches.append(source.read_text(encoding='utf-8'));files[relative]=n.sha(source)
    for relative,(source,digest,binding) in sorted(sources.items()):
        assert n.sha(source)==digest,str(source)
        target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes());assert n.sha(target)==digest
        # Full source additions live in a separate workspace/ review namespace.
        patches.extend(difflib.unified_diff([],target.read_text(encoding='utf-8').splitlines(True),
            fromfile='/dev/null',tofile='b/'+relative))
        files[relative]=digest;manifest.append(dict(file=relative,source=str(source),sha256=digest,binding=binding))
    evidence={
        'research_harness.json':t.OUT/'research_harness.json',
        **{name:t.ROOT/name for name in ('entry_identity_repair.json','temporary_storage_repair.json',
            'preparation_resume.json','receipt_schema_repair.json')},
        'report_packaging_preservation.json':t.ROOT/'report_packaging_repair/preservation.json',
    }
    for name,source in evidence.items():
        relative='provenance/'+name;target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes());files[relative]=n.sha(source)
        assert n.sha(target)==files[relative]
    patch=bundle/'implementation.patch';patch.write_text(''.join(patches),encoding='utf-8')
    files['implementation.patch']=n.sha(patch)
    n.write(bundle/'implementation_manifest.json',dict(files=files,sources=manifest,
        scope='Separate train/oos native diffs plus full authenticated workspace code and exact effective qualifier. Review bundle, not a live activation.',
        qualification_harness_sha256=repair['original_harness_sha256'],receipt_schema_repair_sha256=n.sha(t.ROOT/'receipt_schema_repair.json'),
        reporting_only_packaging_change=True))
    files['implementation_manifest.json']=n.sha(bundle/'implementation_manifest.json')
    return {'review_bundle/'+name:digest for name,digest in files.items()}


def compare(rows):
    return table(['Policy','Window','Return %','Common / old 5m DD %','Complete fine DD %','Native minute-close DD %','Unique entries','Entry days','Entry notional KRW m'],
        [[label,w,num(r[w]['return_pct']),num(r[w]['drawdown_pct']),num(r[w].get('complete_fine_drawdown_pct')),
          num(r[w].get('native_minute_close_drawdown_pct')),r[w]['trades'],r[w]['entry_days'],money(r[w]['entry_notional']) if 'entry_notional' in r[w] else '—']
         for label,r in rows for w in ('train','oos')])


def chart(data):
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    rows=[('Retained selected',data['references']['retained'])]+[(LABELS[r['label']],r) for r in data['candidates']]
    for label,row in rows:
        for j,w in enumerate(('train','oos')):
            folder=Path(row['windows'][w]['folder']);proof=folder/'minute_nav_evidence.json'
            if proof.exists():values=[(r['available_at'],r['equity']) for r in n.read(proof)['common_original_grid']]
            else:
                from datetime import datetime,timedelta
                values=[((datetime.fromisoformat(r[0])+timedelta(minutes=5)).isoformat(),r[1]) for r in n.read(folder/'equity.json')]
            ts=np.array([np.datetime64(x[0][:19]) for x in values]);nav=np.array([x[1] for x in values])/1e8
            axes[0,j].plot(ts,100*(nav-1),label=label,lw=1.2)
            dd=100*(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1)
            axes[1,j].plot(ts,dd,lw=1.2)
    for j,title in enumerate(('IS · Aug 7, 2025–Mar 31, 2026','Inspected OOS · Apr 1–Jul 10, 2026')):
        axes[0,j].set_title(title);axes[0,j].set_ylabel('Cumulative net return (%)');axes[1,j].set_ylabel('Common-grid drawdown (%)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3)
    fig.suptitle('OLR: execution-clock calibration and fixed faster exits',fontsize=16)
    fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(t.OUT/'results.png',dpi=170);plt.close(fig)


def main():
    data=n.read(t.OUT/'results.json');profile=n.read(t.OUT/'selected_research_policy.json');done=n.read(t.ROOT/'qualification_complete.json')
    assert n.sha(t.OUT/'results.json')==profile['results_sha256']==done['results_sha256']
    assert n.sha(t.OUT/'selected_research_policy.json')==done['profile_sha256']
    fine,tested=data['candidates'];refs=data['references'];chart(data)
    body='<h1>OLR fixed minute-exit study</h1>'
    body+=p('Final research policy: '+profile['selected_label']+' ('+profile['selected_id']+'). '+profile['status']+'.')
    body+=p('The full-objective successor gate '+('passed' if tested['full_objective_qualified'] else 'did not pass')+
        '; the separate risk-component gate '+('passed' if tested['risk_component_qualified'] else 'did not pass')+
        '. Risk-component status alone does not authorize promotion. The failed participation-entry pilot is ineligible for combination.')
    body+=p('Only two fixed policies were tested across nominal, combined execution stress and no-upward-compounding IS/OOS: 12 native comparisons. The minute control isolates the source/execution-clock change; E adds the unchanged loss and MFE-fade thresholds on completed minutes after entry day. There was no threshold search and no live activation.')
    body+='<img src="results.png" alt="Cumulative returns and drawdown on the common original completion grid for retained, minute control and E">'
    body+='<h2>Nominal comparison</h2>'+compare([('Original',refs['original']),('Guard',refs['guard']['windows']),('Retained selected',refs['retained']['windows'])]+[(LABELS[r['label']],r['windows']) for r in data['candidates']])
    body+=p('Returns are cumulative and nonannualized, net of native fees and modeled execution costs, on separate KRW100m accounts. Trades are unique filled entry lifecycles; partial exits do not create extra trades. The old policies retain their original five-minute drawdown. New policies are sampled at the same original labels +5 minutes, before a new opening phase. The complete fine-grid maximum additionally includes every minute opening, closing and fill-cost event. Both common and complete fine OOS drawdown must improve against the matched minute control.')
    body+='<h2>What changes returns and drawdown</h2>'
    rows=[]
    for group,label in (('windows','Nominal'),('stress','Combined stress'),('capped','Capped')):
        for w in ('train','oos'):
            old=refs['retained'][group][w];a=fine[group][w];z=tested[group][w]
            rows.append([label,w,num(a['return_pct']-old['return_pct']),num(a['drawdown_pct']-old['drawdown_pct']),
                num(z['return_pct']-a['return_pct']),num(z['drawdown_pct']-a['drawdown_pct']),num(z['complete_fine_drawdown_pct']-a['complete_fine_drawdown_pct']),
                z['trades']-a['trades'],z['entry_days']-a['entry_days']])
    body+=table(['Scenario','Window','Old→minute return pp','Old→minute common DD pp','Minute→E return pp','Minute→E common DD pp','Minute→E full fine DD pp','E entries change','E days change'],rows)
    body+=p('Old-to-minute changes combine corrected source clocks, a finer market-order execution schedule, lagged minute liquidity, and minute observations of held-position excursions. They cannot all be attributed to the faster exit rule. Minute-control-to-E is the matched comparison for faster risk decisions. Negative DD changes mean improvement. Large unrealized profit retreats can drive maximum DD even when final P&L is positive; a stop threshold is not a guaranteed realized-loss cap under gaps, limited capacity and delayed fills.')
    body+='<h2>All registered acceptance gates</h2>'
    for group,label in (('windows','Nominal'),('stress','Combined stress'),('capped','No upward compounding')):
        checks=tested['scenario_gates'][group];risk=tested['risk_component_gates'][group]
        body+=table([label+' gate','Full-objective pass','Also required for risk component'],[[k,'PASS' if v else 'FAIL','Yes' if k in risk else 'No'] for k,v in checks.items()])
        failed=[k for k,v in checks.items() if not v]
        body+=p(label+': '+('all full-objective gates pass.' if not failed else 'fails '+', '.join(failed)+'.'))
    body+=p('Nominal IS preserves original−3pp, guard−3pp, retained−1.5pp and guard DD+2pp. OOS requires retained return+0.25pp and DD−0.75pp, at least186/47 IS/OOS entries and88/29 entry days; net added OOS notional and median added notional each must reach KRW1m. Stress retains original/guard/retained IS gates, OOS return, strictly lower OOS DD and entries/days. Capped retains IS within1.5pp, OOS return, lower OOS DD and entries/days. Every scenario additionally preserves matched fine-control IS within1.5pp and OOS return, lowers both OOS DD measures, and preserves entries/days. Risk-component evaluation removes only frequency/notional gates.')
    body+='<h2>Execution stress and compounding</h2>'
    body+=compare([('Retained — stress',refs['retained']['stress'])]+[(LABELS[r['label']]+' — stress',r['stress']) for r in data['candidates']])
    body+=compare([('Retained — capped',refs['retained']['capped'])]+[(LABELS[r['label']]+' — capped',r['capped']) for r in data['candidates']])
    body+=p('The minute model uses1% of the previous contiguous continuous minute’s volume,0.5% under combined stress, and sourced closing-point volume. It does not borrow volume across gaps or supply first-minute liquidity. Native stressed costs and order/cancel/acknowledgement/partial-fill lifecycle apply to both policies. Capped sizing cannot grow above starting capital after gains, and can fall after losses. Each comparison has its own matching actual IS market-history handoff.')
    body+='<h2>Activation, genuine frequency and preservation of winners</h2>'
    rows=[]
    for group in ('windows','stress','capped'):
        for w in ('train','oos'):
            metric=tested[group][w];m=tested['comparisons'][group][w]['against_fine_control'];e=m['frequency_economics']
            rows.append([group,w,metric['mechanism_activation']['signals'],e['added'],e['removed'],money(e['added_notional']),money(e['net_incremental_notional']),
                money(e['median_added_notional']),money(e['added_net_pnl']),num(m['whole_old_winner_pnl_retention_pct']),m['removed_old_winners'],m['winner_to_loser']])
    body+=table(['Scenario','Window','Minute exit signals','Added entries','Removed entries','Added notional m','Net added notional m','Median added m','Added net P&L m','Whole control-winner P&L retained %','Removed winners','Matched winners→nonwinners'],rows)
    body+=p('Winner retention includes removed control winners as zero and replacement losses as losses. Matched-only retention would hide removed winners. Entry notionals are actual filled quantity × entry price, not candidate counts or proposed orders. Earlier liquidation can change cash, subsequent quantities, history observation membership and later entries; no additional trade is guaranteed by a faster exit.')
    rows=[]
    for group in ('windows','stress','capped'):
        for w in ('train','oos'):
            m=tested['comparisons'][group][w]['against_fine_control']
            rows.append([group,w,money(m['pnl_delta']),money(m['per_unit_execution_effect']),money(m['quantity_effect']),money(m['new_minus_removed_effect']),num(m['top_five_original_winner_pnl_retention_pct'])])
    body+=table(['Scenario','Window','Net P&L delta m','Per-unit effect m','Quantity effect m','Added−removed effect m','Top5 control winners retained %'],rows)
    body+=p('This exact accounting decomposition reconciles to total trade P&L. Per-unit effects include both entry and exit price changes; quantity effects include cash/capacity/compounding responses. It is not a randomized causal decomposition. The top-five figure exposes concentration of preserved or lost winners.')
    is_match=tested['comparisons']['windows']['train']['against_fine_control']
    oo_match=tested['comparisons']['windows']['oos']['against_fine_control']
    stress_match=tested['comparisons']['stress']['oos']['against_fine_control']
    body+='<h2>Why the nominal benefit does not settle qualification</h2>'
    body+=p(f"Nominal IS changes by KRW {money(is_match['pnl_delta'])}m: the per-unit execution term is {money(is_match['per_unit_execution_effect'])}m, the quantity term {money(is_match['quantity_effect'])}m, and added minus removed trades {money(is_match['new_minus_removed_effect'])}m. Faster exits alter available cash and later allocations even when entry rules stay identical. Therefore the IS loss cannot be attributed solely to cutting off winning trades; preserving the exit benefit also requires preserving profitable subsequent exposure.")
    positive=sorted((r for r in oo_match['all_matches'] if r['pnl_delta']>0),key=lambda r:r['pnl_delta'],reverse=True)[:3]
    body+=p(f"The three largest positive nominal OOS trade changes total KRW {money(sum(r['pnl_delta'] for r in positive))}m versus an aggregate change of {money(oo_match['pnl_delta'])}m. Whole control-winner P&L retention is {num(oo_match['whole_old_winner_pnl_retention_pct'])}% nominally and {num(stress_match['whole_old_winner_pnl_retention_pct'])}% under stress. The stress total changes by {money(stress_match['pnl_delta'])}m, split into per-unit {money(stress_match['per_unit_execution_effect'])}m, quantity {money(stress_match['quantity_effect'])}m and added/removed {money(stress_match['new_minus_removed_effect'])}m. The following trade tables expose the concentration and timing differences instead of treating the nominal aggregate as robust alpha.")
    for row in (fine,tested):
        episode=row['windows']['oos']['fine_drawdown_episode']
        body+=p(f"{LABELS[row['label']]}: complete-grid OOS drawdown runs from {episode['peak_at']} at KRW {money(episode['peak_equity'])}m to {episode['trough_at']} at KRW {money(episode['trough_equity'])}m. This is retreat from the highest marked portfolio value, including unrealized gains; it is not the loss measured from starting capital. Earlier exits reduce some losses but do not remove all exposure to overnight gaps, capacity limits or the later sequence of losing positions.")
    for group,label in (('windows','Nominal'),('stress','Combined stress'),('capped','Capped')):
        for w in ('train','oos'):
            matches=tested['comparisons'][group][w]['against_fine_control']['all_matches']
            ranked=sorted(matches,key=lambda r:r['pnl_delta'])
            count=6 if group=='windows' else 3
            selected=ranked[:count]+[r for r in ranked[-count:] if r not in ranked[:count]]
            body+=table([label+' '+w+' entry','Symbol','Control qty','E qty','Control P&L m','E P&L m','Delta m','Control exit','E exit'],
                [[r['entry_decision_time'][:10],r['symbol'],r['old_qty'],r['new_qty'],money(r['old_pnl']),money(r['new_pnl']),money(r['pnl_delta']),r['old_exit_reason'],r['new_exit_reason']] for r in selected])
    body+='<h2>History continuity and source integrity</h2>'
    body+=p('IS begins August7 because of the inherited60-session training period plus one-session embargo, not because older market data are absent. Before April1 each OOS replay restores its own candidate/scenario/capped actual March31 terminal coarse market history, with every observation verified against separately retained OOS source history. Only the verified-flat account resets; market history remains continuous. Older source sessions do not create auction observations that the strategy never routed.')
    rows=[]
    for r in data['candidates']:
        for group in ('windows','stress','capped'):
            a=r[group]['oos']['authentication'];src=a['minute_source_audit']
            rows.append([LABELS[r['label']],group,a['literal_is_terminal_state_restored'],a['verified_prior_observations'],src['consumed_minute_rows'],src['required_symbol_sessions'],src['closing_clock_corrections']['corrected_sessions']])
    body+=table(['Policy','Scenario','Actual IS history restored','Verified prior observations','Independently sourced minute rows','Required symbol-days','Closing-clock corrections verified'],rows)
    body+=p('The new OOS source copy corrects99 closing timestamps across62 names from15:30 to their original15:32 print, preserving exact OHLCV. Old archives and prior studies remain unchanged. Every correction and every consumed minute is independently reparsed inside native qualification. Eighteen omitted basis-repair inputs and one omitted closing-repair archive were recovered against their exact original proof hashes. These are provenance repairs, not newly invented market observations. Failed source attempts and their code are preserved.')
    body+=p('Entry construction canonicalizes only the execution-only minute-exit flag to False in a deep copy for both policies; treatment execution still uses True. This prevents entry artifact hashes from changing simulated auction non-fill draws. Raw control/E entry artifacts are byte-identical, and independent native hashing checks exact retained features, ranks, sizes and rejections. The clock-fix generation separately delivers valid nonexecuting closing observations at their own availability without granting auction capacity. The outer receipt reader was repaired with one exact nested-field substitution after the first control case passed its native audits; that completed case was authenticated and reused. Earlier partial control results had already been observed when these implementation repairs were made. All original generations and failed attempts remain preserved.')
    body+=p('The implementation bundle contains separate train/OOS native patches, the authenticated workspace source-building, entry-identity, audit, resume and temporary-storage modules, and the exact additive receipt-schema repair. Full workspace modules are presented as additions under workspace/ for review. The bundle manifest records each source hash and its frozen or additive binding. The reporting-only packaging correction changes no native result, gate, selected profile or frozen qualification input.')
    body+='<h2>Limits and disposition</h2>'
    body+=p('April–July is repeatedly inspected development data, not independent validation. Sparse routed auction history, fixed universe/taxonomy, inferred archive clocks, minute price proxies and lagged-volume capacity remain. These are not exchange queue or order-book measurements. Source completeness and passing execution audits do not establish future alpha. The retained five-minute policies and fine-clock policies have explicitly different execution assumptions; the matched control is essential.')
    body+=p('The F entry pilot is already complete: nominal IS return44.212% versus53.639% for its retained control, despite OOS13.135% versus12.190%. It failed every IS-return scenario and cannot be combined with E under the registered protocol. No additional parameter search or rejected entry-family union was used to manufacture a passing result.')
    body+=p('Final disposition: '+profile['status']+'. '+('E satisfies every registered full-objective gate.' if tested['full_objective_qualified'] else
        'The original retained selected profile remains the research policy. Any risk-component result is reported separately; it does not satisfy the full objective by itself.'))
    body+='<p><a href="selected_research_policy.json">Selected research profile</a> · <a href="registration.json">Frozen protocol</a> · <a href="review_bundle/implementation.patch">Implementation patch</a> · <a href="results.json">Complete evidence and trade attribution</a></p>'
    html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OLR fixed minute exits</title><style>body{font:16px/1.6 system-ui;color:#182739;max-width:1500px;margin:35px auto;padding:0 24px}h1,h2{line-height:1.2}img{width:100%;height:auto}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}th,td{border:1px solid #ccd6df;text-align:left;padding:7px}th{background:#eaf0f6}a{color:#175490}</style>'+body+'</html>'
    (t.OUT/'report.html').write_text(html,encoding='utf-8')
    class Extract(HTMLParser):
        def __init__(self):super().__init__();self.parts=[]
        def handle_data(self,v):self.parts.append(v.strip())
        def handle_starttag(self,tag,attrs):
            if tag in ('p','h1','h2','tr','td','th'):self.parts.append(' | ' if tag in ('td','th') else '\n')
    parser=Extract();parser.feed(body);(t.OUT/'report.txt').write_text(''.join(parser.parts),encoding='utf-8')
    bundle_files=implementation_bundle()
    n.write(t.OUT/'report_receipt.json',dict(results_sha256=profile['results_sha256'],profile_sha256=done['profile_sha256'],
        files={**{name:n.sha(t.OUT/name) for name in ('report.html','report.txt','results.png')},**bundle_files},script_sha256=n.sha(__file__)))
    print(str(t.OUT/'report.html'))


if __name__=='__main__':main()
