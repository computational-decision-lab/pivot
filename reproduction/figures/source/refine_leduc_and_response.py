"""Minimal label edits and a compact forest plot from saved, verified estimates."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import shutil

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
import numpy as np

TASK = Path(__file__).resolve().parent.parent
WORK = None
FONT = findfont(FontProperties(family='DejaVu Sans', weight='normal'))
BOLD = findfont(FontProperties(family='DejaVu Sans', weight='bold'))

def lines(page):
    return [(''.join(s['text'] for s in line['spans']), line)
            for block in page.get_text('dict')['blocks']
            for line in block.get('lines', [])]

def save_pdf(doc, rel):
    dst = WORK / rel
    tmp = dst.with_suffix('.pdf.tmp')
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp, dst)

def pixel_check(rel, regions):
    old = fitz.open(TASK / 'base' / Path(rel).name)
    new = fitz.open(WORK / rel)
    a = old[0].get_pixmap(dpi=144, alpha=False)
    b = new[0].get_pixmap(dpi=144, alpha=False)
    aa = np.frombuffer(a.samples, dtype=np.uint8).reshape(a.height, a.width, 3)
    bb = np.frombuffer(b.samples, dtype=np.uint8).reshape(b.height, b.width, 3)
    changed = np.any(aa != bb, axis=2)
    allowed = np.zeros(changed.shape, bool)
    for x0, y0, x1, y1 in regions:
        allowed[max(0, int(y0*2)):min(a.height, int(y1*2)+1),
                max(0, int(x0*2)):min(a.width, int(x1*2)+1)] = True
    outside = int((changed & ~allowed).sum())
    assert outside == 0, (rel, outside)
    render = WORK / 'render'
    render.mkdir(parents=True, exist_ok=True)
    new[0].get_pixmap(dpi=180, alpha=False).save(render / (Path(rel).stem + '.png'))
    return {'figure': rel, 'pixels_changed_outside_authorized_regions': outside}

def figure5():
    rel = 'figures/figA_response_footprint.pdf'
    doc = fitz.open(TASK / 'base' / Path(rel).name)
    page = doc[0]
    # Cover the old header without changing shared PDF text cursors or chart paths.
    page.draw_rect(fitz.Rect(0, 0, page.rect.width, 25.5), color=None, fill=(1,1,1))
    page.insert_font(fontname='PIVOTTitleFinal', fontfile=BOLD)
    titles = [
        (44.39625, 10.9088, '(a) Proxy → Actor →'),
        (44.39625, 22.1025, 'Strategic response layers'),
        (235.808, 10.8338, '(b) Strategic − Actor effects'),
        (235.808, 22.1025, 'by opponent family'),
        (425.2948, 10.8338, '(c) Actor vs. Strategic'),
        (425.2948, 22.1025, 'reversal plane'),
    ]
    for x, y, text in titles:
        page.insert_text((x,y), text, fontsize=8.4, fontname='PIVOTTitleFinal')
    annotations = [line for text,line in lines(page)
                   if text in ('Strategic reversal', 'rate: 94.95%')]
    assert len(annotations) == 2
    for line in annotations:
        page.add_redact_annot(fitz.Rect(line['bbox']) + (-.05,-.05,.05,.05),
                             fill=False, cross_out=False)
    page.apply_redactions(images=0, graphics=0, text=0)
    page.insert_font(fontname='PIVOTAnnotationFinal', fontfile=FONT)
    for text,y in [('Strategic reversal', 44.2), ('rate: 94.95%', 51.6)]:
        page.insert_text((430.8,y), text, fontsize=6.8, fontname='PIVOTAnnotationFinal',
                         color=(.22,.22,.22))
    save_pdf(doc, rel)
    return pixel_check(rel, [(0,0,551.1,26), (425,32,493,54)])

def figure6():
    rel = 'figures/additional_leduc/01_proxy_deployment_reversal.pdf'
    doc = fitz.open(TASK / 'base' / Path(rel).name)
    page = doc[0]
    found = [line for text,line in lines(page)
             if text == 'Proxy-positive updates that reverse (%)']
    assert len(found) == 1
    box = fitz.Rect(found[0]['bbox'])
    span = found[0]['spans'][0]
    # Keep the source text cursor intact so the following 2.5% label stays put.
    page.draw_rect(box + (-.05,-.05,.05,.05), color=None, fill=(1,1,1))
    page.insert_font(fontname='PIVOTAxisFinal', fontfile=FONT)
    label = 'Proxy-positive reversals (%)'
    length = fitz.Font(fontfile=FONT).text_length(label, fontsize=span['size'])
    page.insert_text((span['origin'][0], (box.y0+box.y1+length)/2), label,
                     fontsize=span['size'], fontname='PIVOTAxisFinal', rotate=90)
    save_pdf(doc, rel)
    return pixel_check(rel, [(4,5,18,207)])

def figure7():
    rel = 'figures/additional_leduc/05_v4_component_contrasts.pdf'
    source = TASK/'data/leduc/summary.json'
    summary = json.loads(source.read_text())
    labels = [
        ('H_A_allocation_pivot_minus_expected_uniform', 'PIVOT-KG − expected Uniform'),
        ('H_A_registered_draw', 'PIVOT-KG − Uniform draw'),
        ('H_B_acquisition_pivot_minus_ivr_menu', 'PIVOT-KG − IVR menu'),
        ('H_B2_pivot_minus_lucb_fixed', 'PIVOT-KG − LUCB'),
        ('H_B3_pivot_minus_top_proxy_fixed', 'PIVOT-KG − top-proxy'),
        ('H_C_pairing_pivot_minus_unpaired', 'paired − unpaired'),
        ('H_D_exact_minus_noisy_calibration', 'exact − noisy calibration'),
        ('H_D_noisy_calibration_pivot_minus_expected_uniform', 'noisy-calibration PIVOT − Uniform'),
        ('H_E_stop_gain_minus_menu', 'early stopping − full budget'),
        ('H_F_menu_minus_fixed_size', 'menu − fixed query size'),
        ('H_G_interaction_long_minus_short_paired', 'allocation contrast: long − short'),
    ]
    rows = [dict(key=k, label=l, **summary['hypotheses'][k]) for k,l in labels]
    original = fitz.open(TASK/'base'/Path(rel).name)[0]
    blue = (39/255,106/255,158/255)
    # Reconcile each saved interval with the actual existing figure coordinates.
    bars = sorted([d for d in original.get_drawings()
                   if d['type']=='s' and d['color'] and
                   np.allclose(d['color'],blue) and d['rect'].width>0 and
                   d['rect'].height==0], key=lambda d:d['rect'].y0)
    nonzero = [r for r in rows if r['hi'] != r['lo']]
    assert len(bars) == len(nonzero) == 9
    for d,r in zip(bars,nonzero):
        assert np.allclose([d['rect'].x0,d['rect'].x1],
                           [283.92+1423.5*r['lo'],283.92+1423.5*r['hi']],atol=.0002)
    old_numbers = [text for text,line in lines(original) if line['bbox'][0]>400 and text!='Mean']
    assert old_numbers == [f"{r['mean']:+.5f}" for r in rows]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#BCC4CC','axes.labelcolor':'#25323D',
                         'text.color':'#25323D','xtick.color':'#25323D',
                         'ytick.color':'#25323D'})
    fig = plt.figure(figsize=(6.5,4.6),facecolor='white')
    ax = fig.add_axes([.405,.205,.42,.735])
    y = np.arange(len(rows))
    for j,r in enumerate(rows):
        primary = j == 0
        ax.errorbar(r['mean'], j, xerr=[[r['mean']-r['lo']],[r['hi']-r['mean']]],
                    fmt='o',color='#276A9E',ecolor='#276A9E',capsize=3,
                    markersize=6.1 if primary else 5.5,
                    lw=1.7 if primary else 1.3, zorder=4)
        ax.text(1.045,j,f"{r['mean']:+.5f}",transform=ax.get_yaxis_transform(),
                va='center',fontsize=10.1,weight='bold' if primary else 'normal')
    ax.set_yticks(y,[r['label'] for r in rows],fontsize=7.8)
    ax.get_yticklabels()[0].set_weight('bold')
    ax.set_ylim(10.7,-.6)
    ax.set_xlim(-.04,.08)
    ax.set_xticks(np.arange(-.04,.081,.02))
    ax.tick_params(axis='x',labelsize=9.5,labelrotation=35,length=0,pad=7)
    for tick in ax.get_xticklabels(): tick.set_ha('right')
    ax.tick_params(axis='y',length=0,pad=7)
    ax.grid(axis='x',color='#E5E9ED',linewidth=.7)
    ax.set_axisbelow(True)
    ax.axvline(0,color='#25323D',lw=.8,zorder=2)
    ax.set_xlabel('Selected-gain difference\n(positive favors first term)',fontsize=10,labelpad=8)
    ax.text(1.045,1.015,'Mean',transform=ax.transAxes,fontsize=10.1,color='#75818E')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text().strip():
            rect=text.get_window_extent(renderer)
            assert rect.x0>=0 and rect.y0>=0 and rect.x1<=fig.bbox.x1+1 and rect.y1<=fig.bbox.y1+1,text.get_text()
    output=WORK/'figure7';output.mkdir(exist_ok=True)
    for ext in ['pdf','png','svg']:
        metadata={'Creator':'PIVOT saved-evidence figure','CreationDate':None,'ModDate':None} if ext=='pdf' else None
        fig.savefig(output/f'05_v4_component_contrasts.{ext}',dpi=220,metadata=metadata)
    plt.close(fig)
    shutil.copy2(output/'05_v4_component_contrasts.pdf',WORK/rel)
    record={'figure':rel,'saved_summary_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'intervals_verified_against_existing_pdf':9,'means_verified_against_existing_pdf':11,
            'rows':rows,'row_label_font_pt':7.8,'primary_bold_only_first_row':True}
    (output/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    return {k:v for k,v in record.items() if k!='rows'}

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    WORK = args.output.resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    records=[figure5(),figure6(),figure7()]
    (WORK/'leduc-response-audit.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))
