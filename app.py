from flask import Flask, request, send_file, render_template_string
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import calendar
import re
import unicodedata
import pytesseract
from datetime import date
import io
import os
 
app = Flask(__name__)
 
# ── Cores de referência RGB ──────────────────────────────────────
REF_CORES = {
    "Manhã":    np.array([193, 224, 219]),
    "Tarde":    np.array([203, 177, 237]),
    "Noite":    np.array([239, 203, 209]),
    "Descanso": np.array([252, 243, 171]),
    "Férias":   np.array([141, 251, 216]),
    "D":        np.array([134, 155, 230]),
}
MAPA_TURNO = {
    "Manhã": "Manhã", "Tarde": "Tarde", "Noite": "Noite",
    "Descanso": "Descanso", "Férias": "Férias", "D": "Descanso",
}
CORES_TURNOS = {
    "Manhã":    (52,  168, 130),
    "Tarde":    (108,  92, 231),
    "Noite":    (52,   73, 120),
    "Descanso": (245, 200,  80),
    "Férias":   (255, 140,  80),
    "M+T":      (220,  90, 150),
}
MESES_PT = {
    "jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,
    "jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12,
}
DIAS_SEMANA = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
 
GRID_TOP=0.165; GRID_BOTTOM=0.775
HEADER_TOP=0.03; HEADER_BOTTOM=0.14
 
def sem_acentos(txt):
    return ''.join(c for c in unicodedata.normalize('NFD',txt) if unicodedata.category(c)!='Mn')
 
def normalizar(txt):
    return sem_acentos(txt.strip().lower())
 
def extrair_mes_ano(img):
    h,w = img.shape[:2]
    crop = img[int(h*HEADER_TOP):int(h*HEADER_BOTTOM), int(w*0.02):int(w*0.60)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    _,gray = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
    txt = pytesseract.image_to_string(gray, lang="por", config="--psm 6")
    txt_norm = normalizar(txt)
    mes = next((v for k,v in MESES_PT.items() if k in txt_norm), None)
    anos = re.findall(r'20(\d{2})',txt)
    if anos:
        ano = 2000+int(anos[0])
        ano_atual = date.today().year
        if ano < ano_atual-1 or ano > ano_atual+2:
            ano = ano_atual
    else:
        ano = date.today().year
    if not mes:
        raise ValueError(f"Não consegui ler o mês. Texto: {txt!r}")
    return mes, ano
 
def obter_grelha(img):
    h,w = img.shape[:2]
    return img[int(h*GRID_TOP):int(h*GRID_BOTTOM), 0:w]
 
def dividir_grelha(grelha):
    h,w = grelha.shape[:2]
    return [[grelha[int(r*h/6)+4:int((r+1)*h/6)-4,
                    int(c*w/7)+4:int((c+1)*w/7)-4]
             for c in range(7)] for r in range(6)]
 
def extrair_cor_celula(cell):
    h,w = cell.shape[:2]
    zona = cell[int(h*0.15):int(h*0.85), int(w*0.05):int(w*0.95)]
    if zona.size==0: return None,None
    rgb = cv2.cvtColor(zona,cv2.COLOR_BGR2RGB).reshape(-1,3).astype(float)
    hsv = cv2.cvtColor(zona,cv2.COLOR_BGR2HSV).reshape(-1,3).astype(float)
    brilho = rgb.mean(axis=1)
    mask = (brilho>100)&(brilho<235)
    if mask.sum()<20: return None,None
    return np.percentile(rgb[mask],50,axis=0), np.percentile(hsv[mask],50,axis=0)
 
def e_mt(cor_hsv):
    if cor_hsv is None: return False
    return (140<=cor_hsv[0]<=180) and cor_hsv[1]>30
 
def classificar_celula(cell):
    cor_rgb,cor_hsv = extrair_cor_celula(cell)
    if cor_rgb is None: return "?"
    distancias = {n: np.linalg.norm(cor_rgb-r) for n,r in REF_CORES.items()}
    ordenado = sorted(distancias.items(), key=lambda x:x[1])
    melhor_nome,melhor_dist = ordenado[0]
    if melhor_dist>40:
        return "M+T" if e_mt(cor_hsv) else "?"
    turno = MAPA_TURNO[melhor_nome]
    if turno=="Manhã" and e_mt(cor_hsv):
        return "M+T"
    return turno
 
def obter_posicoes_mes(mes,ano):
    fw,nd = calendar.monthrange(ano,mes)
    sc = fw  # 0=Seg ... 6=Dom
    return [(d,(sc+d-1)//7,(sc+d-1)%7) for d in range(1,nd+1)], sc, nd
 
def extrair_turnos(img,mes,ano):
    grelha = obter_grelha(img)
    celulas = dividir_grelha(grelha)
    posicoes,_,_ = obter_posicoes_mes(mes,ano)
    return [(d, classificar_celula(celulas[r][c])) for d,r,c in posicoes]
 
def nome_mes_pt(mes):
    return {1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",
            6:"Junho",7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",
            11:"Novembro",12:"Dezembro"}[mes]
 
def desenhar(resultados,mes,ano):
    COLS=7; PAD=40; CEL_W=140; CEL_H=110; GAP=7
    HEADER_H=90; WEEKDAY_H=34; LEGEND_H=50; RADIUS=12
    BG=(250,250,252); TEXT_DARK=(30,30,40); TEXT_MID=(120,120,135); TEXT_LIGHT=(255,255,255)
 
    def fonte(tamanho, bold=False):
        caminhos = [
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
            f"/usr/share/fonts/truetype/liberation/LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ]
        for p in caminhos:
            try: return ImageFont.truetype(p, tamanho)
            except: continue
        return ImageFont.load_default()
 
    f_titulo  = fonte(38, bold=True)
    f_semana  = fonte(14)
    f_dia_num = fonte(20)
    f_turno   = fonte(16, bold=True)
    f_emoji   = fonte(42)
    f_leg     = fonte(14)
 
    fw,nd = calendar.monthrange(ano,mes)
    sc = (fw+1)%7
    num_rows = ((sc+nd-1)//7)+1
 
    largura = PAD*2 + COLS*CEL_W + (COLS-1)*GAP
    altura  = PAD + HEADER_H + WEEKDAY_H + GAP + num_rows*(CEL_H+GAP) + LEGEND_H + PAD
 
    img  = Image.new("RGB",(largura,altura),BG)
    draw = ImageDraw.Draw(img)
 
    # Título
    draw.text((PAD, PAD+6), f"{nome_mes_pt(mes)} {ano}", fill=TEXT_DARK, font=f_titulo)
 
    # Dias da semana
    y_sem = PAD+HEADER_H
    for c,d in enumerate(DIAS_SEMANA):
        x = PAD+c*(CEL_W+GAP)
        cor = (200,60,80) if c==0 else TEXT_MID
        bbox = draw.textbbox((0,0),d,font=f_semana)
        draw.text((x+(CEL_W-(bbox[2]-bbox[0]))//2, y_sem+8), d, fill=cor, font=f_semana)
 
    draw.line([(PAD,y_sem+WEEKDAY_H),(largura-PAD,y_sem+WEEKDAY_H)], fill=(220,220,228), width=1)
 
    y_grid = y_sem+WEEKDAY_H+GAP
 
    # Células vazias
    for slot in range(sc):
        x = PAD+(slot%(COLS))*(CEL_W+GAP)
        y = y_grid
        draw.rounded_rectangle([x,y,x+CEL_W,y+CEL_H], radius=RADIUS, fill=(240,240,244))
 
    # Células com turnos
    for dia,turno in resultados:
        slot = sc+dia-1
        r,c = slot//7, slot%7
        x = PAD+c*(CEL_W+GAP)
        y = y_grid+r*(CEL_H+GAP)
        cor = CORES_TURNOS.get(turno,(210,210,215))
        draw.rounded_rectangle([x+2,y+2,x+CEL_W+2,y+CEL_H+2], radius=RADIUS, fill=(210,210,218))
        draw.rounded_rectangle([x,y,x+CEL_W,y+CEL_H], radius=RADIUS, fill=cor)
        draw.text((x+12,y+10), str(dia), fill=TEXT_LIGHT, font=f_dia_num)
        if turno!="?":
            label = '😴' if turno=='Descanso' else turno
            f_usar = f_emoji if turno=='Descanso' else f_turno
            bbox = draw.textbbox((0,0),label,font=f_usar)
            tw,th = bbox[2]-bbox[0],bbox[3]-bbox[1]
            draw.text((x+(CEL_W-tw)//2, y+(CEL_H-th)//2+4), label, fill=TEXT_LIGHT, font=f_usar)
 
    # Legenda
    ordem = ["Manhã","Tarde","Noite","Descanso","Férias","M+T"]
    presentes = [t for t in ordem if any(t==tu for _,tu in resultados)]
    y_leg = y_grid+num_rows*(CEL_H+GAP)+14
    lx = PAD
    for chave in presentes:
        cor = CORES_TURNOS[chave]
        draw.ellipse([lx,y_leg+4,lx+14,y_leg+18], fill=cor)
        draw.text((lx+20,y_leg+2), chave, fill=TEXT_MID, font=f_leg)
        bbox = draw.textbbox((0,0),chave,font=f_leg)
        lx += 20+(bbox[2]-bbox[0])+22
 
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf
 
# ── HTML da app ──────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Turnos">
<title>Calendário de Turnos</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
    background: #f5f5f7;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 20px 40px;
  }
  .card {
    background: white;
    border-radius: 22px;
    padding: 32px 28px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
  }
  h1 {
    font-size: 26px;
    font-weight: 700;
    color: #1c1c1e;
    margin-bottom: 6px;
  }
  .sub {
    font-size: 14px;
    color: #8e8e93;
    margin-bottom: 28px;
  }
  .aviso { background:#fff8e1; border-radius:12px; padding:10px 14px; font-size:13px; color:#7a6000; margin-bottom:16px; line-height:1.4; }
  .upload-area {
    border: 2px dashed #d1d1d6;
    border-radius: 16px;
    padding: 36px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: #fafafa;
    position: relative;
  }
  .upload-area:hover, .upload-area.drag { border-color: #007aff; background: #f0f6ff; }
  .upload-area input { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }
  .upload-icon { font-size: 44px; margin-bottom: 12px; }
  .upload-txt { font-size: 16px; font-weight: 600; color: #1c1c1e; margin-bottom: 4px; }
  .upload-sub { font-size: 13px; color: #8e8e93; }
  .preview {
    margin-top: 20px;
    border-radius: 12px;
    overflow: hidden;
    display: none;
  }
  .preview img { width: 100%; border-radius: 12px; }
  .btn {
    margin-top: 20px;
    width: 100%;
    padding: 16px;
    background: #007aff;
    color: white;
    border: none;
    border-radius: 14px;
    font-size: 17px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
    display: none;
  }
  .btn:hover { background: #0062cc; }
  .btn:disabled { background: #b0c8f0; cursor: not-allowed; }
  .loader {
    display: none;
    flex-direction: column;
    align-items: center;
    margin-top: 24px;
    gap: 12px;
  }
  .spinner {
    width: 40px; height: 40px;
    border: 3px solid #e5e5ea;
    border-top-color: #007aff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loader-txt { font-size: 15px; color: #8e8e93; }
  .result { display: none; margin-top: 24px; text-align: center; }
  .result img { width:100%; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.1); }
  .result-btns { display:flex; gap:12px; margin-top:16px; }
  .btn-outline {
    flex:1; padding:14px; border:2px solid #007aff;
    color:#007aff; background:white; border-radius:14px;
    font-size:15px; font-weight:600; cursor:pointer;
  }
  .btn-download {
    flex:1; padding:14px; background:#34c759;
    color:white; border:none; border-radius:14px;
    font-size:15px; font-weight:600; cursor:pointer;
  }
  .error {
    display:none; margin-top:16px; padding:14px;
    background:#fff2f2; border-radius:12px;
    color:#d70015; font-size:14px; text-align:center;
  }
</style>
</head>
<body>
<div class="card">
  <h1>📅 Turnos</h1>
  <p class="sub">Carrega o print do teu calendário</p>
  <div class="aviso">⚙️ Certifica-te que o teu calendário está configurado para começar na <strong>Segunda-feira</strong></div>
 
  <div class="upload-area" id="dropZone">
    <input type="file" id="fileInput" accept="image/*">
    <div class="upload-icon">📸</div>
    <div class="upload-txt">Selecionar imagem</div>
    <div class="upload-sub">PNG, JPG ou HEIC</div>
  </div>
 
  <div class="preview" id="preview">
    <img id="previewImg" src="" alt="Preview">
  </div>
 
  <button class="btn" id="btnGerar" onclick="gerar()">Gerar Calendário</button>
 
  <div class="loader" id="loader">
    <div class="spinner"></div>
    <div class="loader-txt">A processar imagem...</div>
  </div>
 
  <div class="error" id="error"></div>
 
  <div class="result" id="result">
    <img id="resultImg" src="" alt="Calendário">
    <div class="result-btns">
      <button class="btn-outline" onclick="reset()">← Novo</button>
      <button class="btn-download" id="btnDownload">⬇ Guardar</button>
    </div>
  </div>
</div>
 
<script>
let selectedFile = null;
let resultUrl = null;
 
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('previewImg');
const btnGerar = document.getElementById('btnGerar');
const loader = document.getElementById('loader');
const result = document.getElementById('result');
const resultImg = document.getElementById('resultImg');
const error = document.getElementById('error');
const dropZone = document.getElementById('dropZone');
 
fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  preview.style.display = 'block';
  btnGerar.style.display = 'block';
  error.style.display = 'none';
  result.style.display = 'none';
});
 
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event('change')); }
});
 
async function gerar() {
  if (!selectedFile) return;
  btnGerar.disabled = true;
  loader.style.display = 'flex';
  error.style.display = 'none';
  result.style.display = 'none';
 
  const formData = new FormData();
  formData.append('file', selectedFile);
 
  try {
    const resp = await fetch('/processar', { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.erro || 'Erro desconhecido');
    }
    const blob = await resp.blob();
    resultUrl = URL.createObjectURL(blob);
    resultImg.src = resultUrl;
    result.style.display = 'block';
    loader.style.display = 'none';
 
    // Botão de download
    document.getElementById('btnDownload').onclick = () => {
      const a = document.createElement('a');
      a.href = resultUrl;
      a.download = 'calendario_turnos.png';
      a.click();
    };
  } catch(e) {
    error.textContent = '❌ ' + e.message;
    error.style.display = 'block';
    loader.style.display = 'none';
    btnGerar.disabled = false;
  }
}
 
function reset() {
  selectedFile = null;
  fileInput.value = '';
  preview.style.display = 'none';
  btnGerar.style.display = 'none';
  btnGerar.disabled = false;
  result.style.display = 'none';
  error.style.display = 'none';
}
</script>
</body>
</html>"""
 
@app.route("/")
def index():
    return render_template_string(HTML)
 
@app.route("/processar", methods=["POST"])
def processar():
    if "file" not in request.files:
        return {"erro": "Nenhum ficheiro enviado"}, 400
    f = request.files["file"]
    if f.filename == "":
        return {"erro": "Ficheiro inválido"}, 400
    try:
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return {"erro": "Não consegui ler a imagem"}, 400
        mes, ano = extrair_mes_ano(img)
        resultados = extrair_turnos(img, mes, ano)
        buf = desenhar(resultados, mes, ano)
        return send_file(buf, mimetype="image/png",
                         download_name=f"calendario_{mes}_{ano}.png")
    except Exception as e:
        return {"erro": str(e)}, 500
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
