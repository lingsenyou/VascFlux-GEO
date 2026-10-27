"""Recreate Figure S8 and its exact-value addendum from the unlinked CSV only."""
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
MOD = ['endothelial_activation', 'smooth_muscle_contractile', 'inflammation',
       'extracellular_matrix', 'coagulation_thrombosis', 'lipid_handling',
       'oxidative_stress', 'hypoxia_glycolysis']
LAB = ['Endothelial', 'SM contractile', 'Inflammation', 'ECM',
       'Coagulation', 'Lipid handling', 'Oxidative stress', 'Hypoxia/glycolysis']
BLUE, ORANGE, TEAL, INK, GRAY = '#2565A0', '#B96318', '#147D75', '#263746', '#657787'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'axes.labelcolor':INK,
 'text.color':INK,'axes.edgecolor':'#ABB7C1','xtick.color':INK,'ytick.color':INK,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
P = pd.read_csv(ROOT/'small_group_point_values.csv', float_precision='round_trip')

def note(fig, text, y=.04, size=7.3):
    fig.text(.035, y, text, ha='left', va='bottom', fontsize=size, color=GRAY, linespacing=1.45)

def dotplot(ax, context, counts, size=8):
    total = 0
    q = P.loc[P.context.eq(context)]
    pairs = [('Paired change', TEAL, 0.)] if context == 'Exercise training' else [
        ('High grade 4-5', BLUE, -.19), ('Low grade 0-1', ORANGE, .19)]
    for label, color, off in pairs:
        for j, module in enumerate(MOD):
            values = q.loc[q.group.eq(label) & q.module.eq(module), 'value'].to_numpy(float)
            assert len(values) == counts[label]
            jitter = np.linspace(-.12, .12, len(values)) if len(values)>1 else np.array([0.])
            ax.scatter(values, j+off+jitter, s=size, color=color, alpha=.93, edgecolor='white', linewidth=.2, zorder=3)
            ax.plot([values.mean(), values.mean()], [j+off-.105,j+off+.105], color=INK, lw=1.2, zorder=4)
            total += len(values)
    ax.set_yticks(range(8), LAB)
    ax.set_ylim(7.65,-.65)
    ax.axvline(0, color='#B5BEC6',lw=.65,ls='--',zorder=1)
    ax.grid(axis='x', color='#E7ECEF',lw=.5,zorder=0)
    ax.tick_params(axis='y',length=0,pad=3)
    ax.tick_params(axis='x',length=2.5,pad=2)
    ax.set_xlabel('Standardized module value', labelpad=4)
    return total

legend_text = (
    'Figure S8. Individual values underlying existing small-group descriptive summaries. '
    'A shows all five exercise-training paired participants from the 14-pair GSE113873 cohort, '
    'with one value per paired participant and module. Values are existing final-minus-baseline '
    'changes after gene-wise full-series z-scoring (population SD) and median module aggregation. '
    'B-E show all values from the four already estimated GSE281841 restricted contrasts: '
    'peroneal artery (7 high/5 low), posterior tibial artery (6/3), title proxy 0092 (5/4), '
    'and title proxy 0521 (13/4). Each external point is the mean of retained sample module '
    'scores within a literal source record ID and grade group inside that restriction. '
    'Gene z-scoring precedes retained-record exclusions; these descriptive scores are not '
    'training-fitted predictive scores. Literal source IDs are unverified as distinct participants. '
    'Title proxies are not confirmed technical batches. Restrictions reuse records and their '
    'counts cannot be summed. Blue: histological grades 4-5; orange: grades 0-1; teal: paired '
    'exercise change. Black vertical marks are group means; small vertical offsets separate '
    'points for display only. Values are independently sorted within each module, and there '
    'are no cross-module point identifiers. All contrasts are post hoc, unadjusted descriptive '
    'analyses. This addition does not refit models, estimate new intervals, establish participant '
    'independence, or support treatment, transport, clinical or causal inference. The machine-readable '
    'CSV retains the full precision of all 416 displayed values; companion tables list every '
    'value rounded to four decimals.'
)
(ROOT/'FigureS8_legend.txt').write_text(legend_text+'\n',encoding='utf-8')

figure_path = ROOT/'FigureS8_small_group_values.pdf'
drawn = []
with PdfPages(figure_path, metadata={'Title':'Figure S8 - individual values for existing small groups'}) as pdf:
    fig = plt.figure(figsize=(170/25.4,185/25.4))
    fig.text(.035,.97,'Figure S8 | Individual values for small descriptive groups',fontsize=10.2,weight='bold',va='top',color=BLUE)
    fig.text(.035,.91,'A  Exercise training: all five paired participants',fontsize=9.1,weight='bold',va='top')
    fig.text(.035,.875,'GSE113873; five of the 14 paired PAD participants',fontsize=8,va='top',color=GRAY)
    ax=fig.add_axes([.30,.285,.65,.53])
    drawn.append(dotplot(ax,'Exercise training',{'Paired change':5},size=19))
    ax.set_xlabel('Final - baseline module score (full-series standardized units)',fontsize=8)
    note(fig,'Dots: all 5 values per module. Black marks: existing group means.\n'
        'Gene-wise full-series z-scoring, followed by median module aggregation.\n'
        'Vertical offsets separate points only; modules overlap and are not independent replicates.\n'
        'Post hoc descriptive display; no new model fitting, tests or interval estimates.',y=.09,size=8)
    pdf.savefig(fig)
    fig.savefig(ROOT/'FigureS8_page1.png',dpi=220)
    plt.close(fig)

    fig=plt.figure(figsize=(170/25.4,185/25.4))
    fig.text(.035,.975,'Figure S8 | Continued: GSE281841 source record values',fontsize=10,weight='bold',va='top',color=BLUE)
    config=[('Peroneal artery','B',7,5,.245,.59,.035,.925),
            ('Posterior tibial artery','C',6,3,.745,.59,.535,.925),
            ('Title proxy 0092','D',5,4,.245,.175,.035,.51),
            ('Title proxy 0521','E',13,4,.745,.175,.535,.51)]
    for context,letter,nh,nl,x,y,tx,ty in config:
        fig.text(tx,ty,f'{letter}  {context}',fontsize=8.4,weight='bold',va='top')
        fig.text(tx,ty-.032,f'{nh} high-grade / {nl} low-grade source record IDs',fontsize=6.9,va='top',color=GRAY)
        ax=fig.add_axes([x,y,.218,.275])
        drawn.append(dotplot(ax,context,{'High grade 4-5':nh,'Low grade 0-1':nl},size=7.5))
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel('Module score',fontsize=7)
    fig.legend(handles=[Line2D([],[],ls='',marker='o',ms=3.5,color=BLUE,label='Grades 4-5'),
                        Line2D([],[],ls='',marker='o',ms=3.5,color=ORANGE,label='Grades 0-1')],
               loc='lower left',bbox_to_anchor=(.025,.077),ncol=2,frameon=False,fontsize=7.5)
    note(fig,'Scores averaged within literal source record ID inside each restriction; identity is unconfirmed.\n'
        'Restrictions overlap; counts cannot be summed. Title proxies are not confirmed batches.\n'
        'All points shown; black marks are existing means. Post hoc descriptive comparisons only.',y=.015,size=6.9)
    pdf.savefig(fig)
    fig.savefig(ROOT/'FigureS8_page2.png',dpi=220)
    plt.close(fig)
assert drawn == [40,96,72,72,136] and sum(drawn)==416

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='SmallBody',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=7,textColor=colors.HexColor(INK)))
styles.add(ParagraphStyle(name='SmallCell',fontName='Helvetica',fontSize=7.2,leading=9.3,textColor=colors.HexColor(INK)))
styles.add(ParagraphStyle(name='SmallHeading',fontName='Helvetica-Bold',fontSize=11,leading=14,spaceAfter=9,textColor=colors.HexColor(BLUE)))
styles.add(ParagraphStyle(name='TableHeading',fontName='Helvetica-Bold',fontSize=9,leading=12,spaceAfter=6,textColor=colors.HexColor(INK)))
tables_path=ROOT/'_point_tables_internal.pdf'
doc=SimpleDocTemplate(str(tables_path),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=17*mm,bottomMargin=17*mm,
                      title='VascFlux-GEO small-group point-value reporting addendum')
story=[Paragraph('Figure S8: reporting addendum and complete point values',styles['SmallHeading']),
       Paragraph(legend_text,styles['SmallBody']),
       Paragraph('Every value is listed below to four decimal places. The CSV supplies full precision. '
                 'Rows within each module are sorted independently; the lists cannot be used to link values across modules.',styles['SmallBody'])]

contexts=['Exercise training','Peroneal artery','Posterior tibial artery','Title proxy 0092','Title proxy 0521']
for index,context in enumerate(contexts):
    if index in [1,3]:
        story.append(PageBreak())
    story.append(Paragraph(f'Point-value table {index+1}. {context}',styles['TableHeading']))
    q=P.loc[P.context.eq(context)]
    labels=['Paired change'] if index==0 else ['High grade 4-5','Low grade 0-1']
    header=['Module']+[f'{label} (n={len(q.loc[q.group.eq(label) & q.module.eq(MOD[0])])})' for label in labels]
    data=[[Paragraph(h,styles['SmallCell']) for h in header]]
    for module,label in zip(MOD,LAB):
        row=[Paragraph(label,styles['SmallCell'])]
        for group in labels:
            vals=q.loc[q.group.eq(group) & q.module.eq(module),'value'].to_numpy(float)
            row.append(Paragraph(', '.join(f'{v:.4f}' for v in vals),styles['SmallCell']))
        data.append(row)
    widths=[34*mm,140*mm] if index==0 else [34*mm,70*mm,70*mm]
    table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF1F7')),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#718391')),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#718391')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8FAFB')]),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(table)
    story.append(Spacer(1,9*mm))

def footer(canvas,doc):
    canvas.setFont('Helvetica',7)
    canvas.setFillColor(colors.HexColor(GRAY))
    canvas.drawString(18*mm,10*mm,'Local review addendum - post hoc descriptive values; no original identifiers')
    canvas.drawRightString(192*mm,10*mm,str(doc.page))
doc.build(story,onFirstPage=footer,onLaterPages=footer)
combined=PdfWriter()
combined.append(str(figure_path))
combined.append(str(tables_path))
addendum_path=ROOT/'VascFluxGEO_small_group_reporting_addendum.pdf'
with addendum_path.open('wb') as handle:
    combined.write(handle)
tables_path.unlink()
readme=(
    'LOCAL REVIEW ADDENDUM - NOT UPLOADED OR PUBLISHED\n\n'
    'Figure S8 has two pages and panels A-E. Its companion addendum includes the complete legend '
    'and every point value rounded to four decimals. The CSV contains all 416 exact exported values. '
    'The exercise group has five paired participants. External values use literal source record IDs '
    'whose identity as distinct participants is unconfirmed. Restrictions reuse records.\n\n'
    'This adds no new model training, tests, interval estimates or manuscript conclusions. '
    'Values were independently sorted within each context, group and module. No original subject, '
    'sample or source IDs, ages or cross-module point identifiers are exported. This is not a '
    'de-identification certificate or publication consent determination.\n\n'
    'Recreate the two PDFs and PNGs by running python plot_reporting_points.py in this directory '
    'with Python 3.11.9, NumPy 2.4.6, pandas 2.3.3 and Matplotlib 3.10.9; reportlab and pypdf '
    'are also used for document assembly. No linked source CSV is required for redrawing.\n\n'
    'No software licence, repository URL, legal acceptance or public release is established by '
    'this local addendum. Existing manuscript and submission attachments were not modified.\n')
(ROOT/'README_REPORTING_ADDENDUM.txt').write_text(readme,encoding='utf-8')
(ROOT/'figure_display_validation.json').write_text(json.dumps({
    'plotted_individual_point_counts_A_to_E':drawn,'all_416_values_plotted':True,
    'figure_pages':len(PdfReader(figure_path).pages),'addendum_pages':len(PdfReader(addendum_path).pages),
    'full_legend_present':True,'all_point_tables_included':5,
    'table_precision_decimals':4,'CSV_full_precision':True},indent=2)+'\n',encoding='utf-8')
