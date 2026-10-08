# ############################################################################
# STALE — DO NOT RUN. `pipeline_flowchart.svg` IS THE SOURCE OF TRUTH.
#
# As of 2026-08-20 this generator NO LONGER reproduces manuscript Figure 2. The
# SVG has since been hand-edited in Inkscape, and those edits were never folded
# back here. Running this script would OVERWRITE pipeline_flowchart.svg and
# silently revert:
#   * the model name  -> this file still says "CAPE-MPNN-II", two renames out of
#     date (the figure was "DEIMUN-MPNN" for a while; correct is "DeImmu-MPNN")
#   * the Panel B numbers -> this file still hardcodes the pre-re-anchor
#     -32.7% / -39.7% (PWM, old operating point). Correct, netMHCIIpan on the
#     deployed model f16b51e6: monomer -37.9%, binder panel -50.4%.
#
# To change Figure 2: edit `pipeline_flowchart.svg` (Inkscape or the <tspan>
# text directly), then rasterize at the manuscript's width:
#   inkscape pipeline_flowchart.svg --export-type=png \
#     --export-filename=pipeline_overview_figure_vN_<label>.png \
#     --export-width=4375 --export-background=white
#   cp pipeline_overview_figure_vN_<label>.png pipeline_overview_figure.png
# Keep the versioned render; only the generic staged name is refreshed.
#
# Retained for the layout/geometry reference and the PIL text-fit check below.
# ############################################################################
#
# Generate the DPO pipeline flowchart (manuscript Figure 2) as SVG + HTML.
# Run from figures/:  python make_pipeline_flowchart.py
# Then rasterize the SVG to the PNG the manuscript embeds:
#   inkscape pipeline_flowchart.svg --export-type=png \
#       --export-filename=pipeline_overview_figure.png --export-width=2400 --export-background=white
#
# v2 (2026-08-07): (1) corrected Panel B monomer number -39% -> -32.7% deployed
#   (-34.8% without the acidic-composition regularizer), matching the abstract;
#   (2) fonts enlarged ~1.5x relative to the canvas width so on-page text is legible.
# v3 (2026-08-07): Helvetica/Arial are NOT installed, so inkscape was substituting the
#   wider DejaVu Sans -> text overflowed boxes. Switched font to Liberation Sans (installed,
#   Arial-metric width) and added a measured fit-check (PIL) that FAILS the build if any
#   line exceeds its box; repositioned the "regenerate pairs" loop label out of the
#   candidate-sequence-pairs box.
import html
from PIL import ImageFont

FONT = "Liberation Sans, Arial, sans-serif"
# metric-compatible TTFs actually used at render time, for the fit-check
FT_REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FT_BLD = "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
PAD = 14  # required inner horizontal padding (viewBox units) each side

W, H = 1400, 920
BLUE="#2c6fbb"; DBLUE="#1f4e79"; ORANGE="#e08a1e"; GRAY="#8a9099"
GREENF="#dff0e4"; GREENE="#2e7d32"; INK="#2b2b2b"; NEU="#eef1f5"
parts=[]
_boxes=[]   # (x,y,w,h,lines,fs,bold) for validation

def esc(s): return html.escape(s)

def _textw(s, fs, bold):
    f = ImageFont.truetype(FT_BLD if bold else FT_REG, int(round(fs)))
    return f.getlength(s)

def box(x,y,w,h,lines,fill,stroke,tcolor,fs=23,rx=12,bold=False,sw=1.8):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    n=len(lines); lh=fs*1.24; y0=y+h/2-(n-1)*lh/2
    fw="700" if bold else "400"
    for i,ln in enumerate(lines):
        parts.append(f'<text x="{x+w/2}" y="{y0+i*lh}" text-anchor="middle" '
                     f'dominant-baseline="central" font-size="{fs}" font-weight="{fw}" '
                     f'fill="{tcolor}">{esc(ln)}</text>')
    _boxes.append((x,y,w,h,lines,fs,bold))

def label(x,y,s,fs=20,c=INK,bold=True,anchor="middle"):
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" '
                 f'font-weight="{"700" if bold else "400"}" fill="{c}">{esc(s)}</text>')

def arr(x1,y1,x2,y2,c=INK,sw=2.6,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{c}" stroke-width="{sw}" '
                 f'fill="none" marker-end="url(#ah)"{d}/>')

def curve(dpath,c=INK,sw=2.6,dash=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<path d="{dpath}" stroke="{c}" stroke-width="{sw}" fill="none" '
                 f'marker-end="url(#ah)"{d}/>')

# ================= Panel A: DPO training loop =================
xa=330
label(48,52,"A",32,anchor="start")
box(210,80,240,58,["PDB backbone"],NEU,INK,INK)
arr(xa,138,xa,182)
box(190,182,280,64,["ProteinMPNN policy"],BLUE,DBLUE,"white",bold=True)
arr(xa,246,xa,292)
box(170,292,320,60,["candidate sequence pairs"],NEU,INK,INK)
# three scorers
arr(xa,352,150,410); arr(xa,352,xa,410); arr(xa,352,520,410)
box(50,410,200,80,["MHC class I","PWM"],ORANGE,"#a5641a","white",fs=21)
box(260,410,180,80,["MHC class II","PWM"],BLUE,DBLUE,"white",fs=21)
box(450,410,210,80,["acidic-composition","regularizer (λ)"],GRAY,"#5f656d","white",fs=19)
arr(150,490,300,548); arr(xa,490,xa,548); arr(555,490,360,548)
box(190,548,280,64,["preference score"],DBLUE,"#12385c","white",bold=True)
arr(xa,612,xa,656)
box(190,656,280,64,["DPO gradient update"],DBLUE,"#12385c","white",bold=True)
# loop back to policy (orthogonal, pinned at x=36 to clear the scorer boxes)
curve("M190,688 L58,688 Q36,688 36,666 L36,236 Q36,214 58,214 L185,214",
      c=BLUE,sw=2.8,dash="8 5")
# loop label: clear left strip (x<170, above the scorer row), left-anchored at x=48 and
# fs14 so its widest line (~109 px) ends near x=157, clearing the candidate box (x=170)
# and the orange MHC-I box (y>=410) below.
_LBL_X=48; _LBL_FS=14
_lbl=["regenerate pairs","every 2 epochs","× 200 epochs"]
for i,s in enumerate(_lbl):
    label(_LBL_X, 300+i*22, s, _LBL_FS, BLUE, bold=True, anchor="start")
_lbl_right=_LBL_X+max(_textw(s,_LBL_FS,True) for s in _lbl)
assert _lbl_right < 168, f"loop label overruns candidate box: right={_lbl_right:.1f}"

# divider
parts.append(f'<line x1="695" y1="70" x2="695" y2="{H-40}" stroke="#cdd3da" '
             f'stroke-width="1.6" stroke-dasharray="5 5"/>')

# ================= Panel B: evaluation =================
xL, xR = 905, 1245
label(748,52,"B",32,anchor="start")
box(830,90,470,64,["CAPE-MPNN-II  (fine-tuned model)"],"#1b5e9b","#12385c","white",bold=True,fs=24)
arr(1065,154,xL,214); arr(1065,154,xR,214)
# left branch: monomers
box(760,214,290,58,["100 held-out","monomer proteins"],NEU,INK,INK,fs=20)
arr(xL,272,xL,318)
box(735,318,340,104,["predicted MHC class II","burden  −32.7%","(−34.8% without","λ regularizer)"],
    GREENF,GREENE,INK,bold=True,fs=22)
arr(xL,422,xL,466)
box(775,466,260,56,["confirmed vs","netMHCIIpan"],NEU,INK,INK,fs=19)
# right branch: binders
box(1105,214,280,58,["11 de novo","binder complexes"],NEU,INK,INK,fs=20)
arr(xR,272,xR,318)
box(1120,318,250,72,["redesign binder chain,","target held fixed"],BLUE,DBLUE,"white",fs=20)
arr(xR,390,xR,436)
box(1110,436,270,74,["burden  −39.7%","(11/11 complexes)"],GREENF,GREENE,INK,bold=True,fs=22)
arr(xR,510,xR,556)
box(1095,556,300,72,["binding retained","(Boltz-2, AF2 initial-guess)"],GREENF,GREENE,INK,fs=19)

# ---- fit-check: fail loudly if any line overflows its box ----
bad=[]
for (x,y,w,h,lines,fs,bold) in _boxes:
    avail=w-2*PAD
    for ln in lines:
        wln=_textw(ln,fs,bold)
        if wln>avail:
            bad.append((ln, round(wln,1), round(avail,1), w))
if bad:
    print("OVERFLOW (line, width_px, avail_px, box_w):")
    for b in bad: print("  ", b)
    raise SystemExit("Fix box widths/fonts before shipping.")
print(f"fit-check OK — {len(_boxes)} boxes, all lines within box width")

svg=(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
     f'font-family="{FONT}">'
     f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
     f'markerHeight="7" orient="auto-start-reverse">'
     f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>'
     f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>'
     + "".join(parts) + '</svg>')

open("pipeline_flowchart.svg","w").write(svg)
open("pipeline_flowchart.html","w").write(
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<title>DPO pipeline flowchart</title></head><body>"
    "<h1>Immunogenicity-aware ProteinMPNN: training and evaluation</h1>"
    + svg + "</body></html>")
print("wrote pipeline_flowchart.svg / .html")
