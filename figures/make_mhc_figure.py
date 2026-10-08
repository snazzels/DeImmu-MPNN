import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import os

import os as _os
ICON=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"icons")
BLUE="#2c6fbb"; GRAY="#9098a1"; RED="#c0392b"; DARK="#2b2b2b"
def L(n): return plt.imread(os.path.join(ICON,n+".png"))
def place(ax,name,x,y,w):
    img=L(name); ih,iw=img.shape[:2]; h=w*ih/iw
    iax=ax.inset_axes([x-w/2,y-h/2,w,h],transform=ax.transData)
    iax.imshow(img); iax.axis("off"); iax.patch.set_alpha(0)
def arrow(ax,p0,p1,color=DARK,lw=2.2,rad=0.0,ls="-"):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=18,lw=lw,
        color=color,connectionstyle=f"arc3,rad={rad}",linestyle=ls,shrinkA=3,shrinkB=3,zorder=3))
def txt(ax,x,y,s,size=11,w="normal",c=DARK,ha="center"):
    ax.text(x,y,s,fontsize=size,fontweight=w,color=c,ha=ha,va="center",zorder=6)

fig,ax=plt.subplots(figsize=(17,8.5))
ax.set_xlim(0,17); ax.set_ylim(0,8.5); ax.set_aspect("equal"); ax.axis("off")

# Figure title removed per manuscript style (description lives in the caption); lane labels retained.

# ---- therapeutic protein ----
ax.add_patch(Circle((1.0,4.5),0.40,facecolor="#b9c6d6",edgecolor=DARK,lw=1.5,zorder=4))
for dx,dy in [(-0.14,0.11),(0.15,0.05),(-0.02,-0.16),(0.11,-0.11)]:
    ax.add_patch(Circle((1.0+dx,4.5+dy),0.15,facecolor="#93a7be",edgecolor=DARK,lw=0.9,zorder=5))
txt(ax,1.0,3.75,"de novo\ntherapeutic protein",10,"bold")
arrow(ax,(1.5,4.5),(2.15,4.5),lw=2.4)
txt(ax,1.82,4.78,"uptake",8.5,c=GRAY)

# ---- APC ----
place(ax,"dendritic-cell-1",3.15,4.5,2.15)
txt(ax,3.15,3.05,"antigen-presenting cell",10,"bold")
arrow(ax,(4.15,5.2),(5.15,6.55),color=GRAY,lw=2.2,rad=0.15)
arrow(ax,(4.15,3.8),(5.2,3.05),color=BLUE,lw=2.6,rad=-0.15)

# =============== MHC class I lane (muted) ===============
yI=6.6
txt(ax,9.6,7.55,"MHC class I  —  cytosolic / proteasomal  (CD8$^+$; not the ADA driver)",12,"bold",c=GRAY)
place(ax,"proteasome",6.0,yI,1.1); txt(ax,6.0,yI-0.95,"proteasome",9,c=GRAY)
arrow(ax,(6.6,yI),(7.45,yI),color=GRAY)
place(ax,"mhc1",8.05,yI,0.95); txt(ax,8.05,yI-0.95,"MHC I · 8–10-mer",9,c=GRAY)
arrow(ax,(8.6,yI),(9.45,yI),color=GRAY)
place(ax,"t-lymphocyte",10.25,yI,1.3); txt(ax,10.25,yI-0.95,"CD8$^+$ cytotoxic T cell",9,c=GRAY)
arrow(ax,(11.0,yI),(11.9,yI),color=GRAY)
txt(ax,13.0,yI,"cytotoxic\nT-cell response",10,"normal",c=GRAY)

# =============== MHC class II lane (emphasized) ===============
yII=2.9
ax.add_patch(Circle((6.0,yII),0.52,facecolor="none",edgecolor=BLUE,lw=2.2,zorder=4))
ax.add_patch(Circle((6.0,yII),0.40,facecolor="#dbe8f5",edgecolor=BLUE,lw=1.0,zorder=3))
txt(ax,6.0,yII-0.9,"endosome / lysosome",9,c=BLUE)
arrow(ax,(6.6,yII),(7.45,yII),color=BLUE)
place(ax,"MHC2",8.05,yII,0.95); txt(ax,8.05,yII-0.9,"MHC II · 13–25-mer",9,c=BLUE)
arrow(ax,(8.6,yII),(9.4,yII),color=BLUE)
place(ax,"t-lymphocyte",10.1,yII,1.2); txt(ax,10.1,yII-0.9,"CD4$^+$ T-helper",9,c=BLUE)
arrow(ax,(10.8,yII),(11.55,yII),color=BLUE)
place(ax,"b-lymphocyte",12.2,yII,1.2); txt(ax,12.2,yII-0.9,"B cell",9,c=BLUE)
arrow(ax,(12.9,yII),(13.65,yII),color=BLUE)
place(ax,"plasma_cell",14.3,yII,1.4); txt(ax,14.3,yII-0.9,"plasma cell",9,c=BLUE)

# --- right-margin ADA motif: plasma cell -> antibody -> neutralized therapeutic ---
arrow(ax,(14.7,3.5),(15.35,4.25),color=RED,rad=0.0,lw=2.2)
place(ax,"immunoglobulin",15.5,4.95,0.95)
txt(ax,16.4,4.95,"anti-drug\nantibody\n(ADA)",9.5,"bold",c=RED)
arrow(ax,(15.5,5.6),(15.5,6.15),color=RED,rad=0.0,lw=2.2)
# small "neutralized" therapeutic being bound
ax.add_patch(Circle((15.5,6.6),0.30,facecolor="#b9c6d6",edgecolor=DARK,lw=1.3,zorder=4))
for dx,dy in [(-0.11,0.08),(0.11,0.03),(0.0,-0.12)]:
    ax.add_patch(Circle((15.5+dx,6.6+dy),0.12,facecolor="#93a7be",edgecolor=DARK,lw=0.8,zorder=5))
txt(ax,15.5,7.25,"therapeutic\nneutralized",9.5,"bold",c=RED)

txt(ax,8.2,0.95,"MHC class II  —  endosomal / lysosomal  (CD4$^+$ → B cell → antibody)",12,"bold",c=BLUE)

plt.savefig(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"mhc_mechanism_figure.png"),dpi=300,bbox_inches="tight",facecolor="white")
print("wrote mhc_mechanism_figure.png")
