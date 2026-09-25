"""Extract saved estimates; no fitting, resampling, or experiment execution."""
import csv, json, hashlib, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SERVER = Path(__import__('os').environ.get('LEDUC_SOURCE_ROOT', 'source-results'))
sources = []
def get(rel, name):
    src = SERVER/rel
    dst = ROOT/'source_summaries'/name
    shutil.copy2(src, dst)
    sources.append({'source_server_path':'/root/autodl-tmp/'+rel,'file':str(dst.relative_to(ROOT)),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
    return json.loads(dst.read_text())
v4=get('openspiel_v4_20260921/leduc_v4_confirm/analysis/summary.json','v4_confirm_summary.json')
rows=[]
def row(figure, key, label, mean, lo='', hi='', n='', version='', **extra):
    rows.append(dict(figure=figure,key=key,label=label,mean=mean,lo=lo,hi=hi,n=n,version=version,**extra))
for h,label in [('1','Short (1 step)'),('8','Long (8 steps)')]:
    b=v4['mechanism'][h]['IRR']; row('01_reversal',h,label,**b)
labels={
'proxy_only':'Proxy only (0 HF hands)',
'no_hf':'Calibrated, no HF (0 hands)',
'uniform_fixed':'Uniform fixed (exact expectation)',
'uniform_small':'Uniform small (MC expectation)',
'pivot_kg_menu':'PIVOT-KG menu',
'pivot_kg_menu_stop':'PIVOT-KG menu + stopping',
'ivr_menu':'IVR menu',
'lucb_fixed':'LUCB fixed',
'ivr_fixed':'IVR fixed',
'pivot_kg_fixed':'PIVOT-KG fixed',
'pivot_kg_menu_unpaired':'PIVOT-KG menu, unpaired',
'top_proxy_fixed':'Top-proxy fixed',
'all_hf_fixed':'All-HF fixed (65,536 hands)'}
order=['proxy_only','no_hf','uniform_fixed','uniform_small','pivot_kg_menu','pivot_kg_menu_stop','ivr_menu','lucb_fixed','ivr_fixed','pivot_kg_fixed','pivot_kg_menu_unpaired','top_proxy_fixed','all_hf_fixed']
table={r['method']:r for r in v4['method_tables']['h8|cap16384|exact']}
assert set(table)==set(order)
for m in order:
    r=table[m];row('02_method_regret',m,labels[m],r['mean_isr'],r['isr_lo'],r['isr_hi'],r['n'],mean_gain=r['mean_gain'],mean_hands=r['mean_hands'])
for m,label in [('pivot_kg_menu','Full budget'),('pivot_kg_menu_stop','Early stopping')]:
    r=table[m];row('03_stopping',m,label,r['mean_hands'],n=r['n'],mean_gain=r['mean_gain'])
points=['eta_0.05','eta_0.10','eta_0.25','eta_0.40','pex_0.80','pex_0.95','mix_cont']
for point in points:
    for version,folder in [('Frozen','analysis'),('Corrected','analysis_corrected')]:
        d=get('openspiel_v2_20260918/leduc_v3_suite/'+point+'_confirm/'+folder+'/summary.json','E1_'+point+'_'+version.lower()+'.json')
        h=d['hypotheses']['H2_long_primary_gain_minus_comparator_cap192']
        label=point.replace('eta_',r'$\eta = ')+'$' if point.startswith('eta_') else (r'$P(\mathrm{exploiter}) = '+point[4:]+'$' if point.startswith('pex_') else 'Continuous mixture')
        row('04_e1',point,label,h['mean'],h['lo'],h['hi'],h['n'],version=version)
contrasts=[('H_A_allocation_pivot_minus_expected_uniform','H_A: menu - expected Uniform'),('H_A_registered_draw','H_A: menu - registered Uniform draw'),('H_B_acquisition_pivot_minus_ivr_menu','H_B: menu - IVR menu'),('H_B2_pivot_minus_lucb_fixed','H_B2: menu - LUCB fixed'),('H_B3_pivot_minus_top_proxy_fixed','H_B3: menu - top-proxy fixed'),('H_C_pairing_pivot_minus_unpaired','H_C: paired - unpaired menu'),('H_D_exact_minus_noisy_calibration','H_D: exact - noisy calibration'),('H_D_noisy_calibration_pivot_minus_expected_uniform','H_D: noisy menu - expected Uniform'),('H_E_stop_gain_minus_menu','H_E: stopping - full menu'),('H_F_menu_minus_fixed_size','H_F: menu - fixed size'),('H_G_interaction_long_minus_short_paired','H_G: long - short allocation contrast')]
for k,label in contrasts:
    h=v4['hypotheses'][k];row('05_contrasts',k,label,h['mean'],h['lo'],h['hi'],h['n'],signflip_p=h.get('signflip_p',''))
fields=['figure','key','label','mean','lo','hi','n','version','mean_gain','mean_hands','signflip_p']
with (ROOT/'plot_data.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader();w.writerows(rows)
(ROOT/'source_manifest.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
print('Saved',len(rows),'rows;',len(sources),'source summaries')
