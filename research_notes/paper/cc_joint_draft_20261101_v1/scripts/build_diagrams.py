"""Original schematic and source-matrix plots. Does not import the simulator."""
from pathlib import Path
import ast, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})

def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5.2));ax.set_xlim(0,10);ax.set_ylim(0,7);ax.axis('off')
def box(x,y,w,h,text,color='#EAF3F8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#37647A',lw=1.1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b,color='#37647A',rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.2,color=color,connectionstyle=f'arc3,rad={rad}'))
box(.1,5.6,3.0,1.1,'Arrived jobs\nObserved progress\nBatch / checkpoint')
box(3.55,5.6,2.7,1.1,'Sia GPU candidates\nHeterogeneity\nand elasticity')
box(6.75,5.6,3.05,1.1,'R0 + Synergy shape\nSynthetic CPU / RAM\nresponse','#FFF1D5')
box(.1,3.5,3,1.15,'Ordered admission\nHeld service\nand FIFO head')
box(3.55,3.5,2.7,1.15,'Joint column MIP\nGPU / quotas\nand placement')
box(6.75,3.5,3.05,1.15,'Current ETA scoring\nStartup cost\nand remaining work')
box(.1,1.05,3,1.25,'Productive service ledger\nactual completion events')
box(3.55,1.05,2.7,1.25,'Atomic validation\nCapacity\nand complete locks')
box(6.75,1.05,3.05,1.25,'Sia training execution\nRaw rate × rho once\nDelay kept separate','#E5F2EA')
for a,b in [((3.1,6.15),(3.5,6.15)),((4.9,5.55),(4.9,4.7)),((8.25,5.55),(8.25,4.7)),((6.7,4.05),(6.3,4.05)),((3.1,4.05),(3.5,4.05)),((4.9,3.45),(4.9,2.35)),((6.3,1.65),(6.7,1.65)),((6.75,1.04),(3.1,1.04)),((1.6,2.35),(1.6,3.45)),((1.6,5.55),(1.6,4.7))]:arrow(a,b)
ax.text(5,.18,'Observed-state feedback; no future completion or arrival information',ha='center',fontsize=10,color='#444444')
save(fig,'fig01_architecture')

cpu=np.array([1,2,3,4,5,6,9,12,24]);ram=np.array([20.83,62.5,125,187.5,250])
templates=[('alexnet_1.py','AlexNet shape','#0072B2'),('deepspeech_1.py','DeepSpeech shape','#D55E00'),('bert_1.py','BERT flat shape','#009E73')]
source=[]
fig,axs=plt.subplots(1,2,figsize=(8,3.7))
for filename,label,color in templates:
    path=PROJECT/'research_sources/Synergy/simulator/models'/filename
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    assigns=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='tput' for t in n.targets)]
    m=np.asarray(ast.literal_eval(assigns[0].value.args[0]),dtype=float)
    assert m.shape==(9,5)
    b=m[2,1]
    axs[0].plot(cpu,np.sqrt(m[:,2]/b),marker='o',color=color,label=label,markersize=4)
    axs[1].plot(ram,np.sqrt(m[7,:]/b),marker='o',color=color,label=label,markersize=4)
    source.append({'template':filename,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest().upper(),'matrix':m.tolist(),'reference':float(b)})
for ax in axs:
    ax.axhline(1,ls='--',lw=.8,color='#777777');ax.grid(alpha=.2);ax.set_ylim(0,1.95)
    ax.set_ylabel('Relative response (alpha = 0.5)')
axs[0].set_title('(a) CPU variation\nsource RAM fixed at 125 GB',fontsize=11);axs[0].set_xlabel('Source CPU coordinate')
axs[1].set_title('(b) RAM variation\nsource CPU fixed at 12',fontsize=11);axs[1].set_xlabel('Source RAM coordinate (GB)')
axs[1].legend(loc='lower right',fontsize=8.8,frameon=False)
fig.tight_layout(w_pad=1)
save(fig,'fig06_response_templates')
(OUT/'diagram_sources.json').write_text(json.dumps({'source_matrices':source,'reference_cpu_ram':[3,62.5],'note':'Raw source-template axes. No phi applied; not target-hardware measurements. Neutral not plotted.'},ensure_ascii=False,indent=2),encoding='utf-8')
print('Created architecture and 3-template response plots; no simulation.')
