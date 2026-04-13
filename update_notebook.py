import json

with open('notebooks/tech_challenge_fase1.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# CELL 21: SHAP em portugues
nb['cells'][21]['source'] = """X_test_np = X_test_s.values; feat_list = X_test_s.columns.tolist()
explainer = shap.TreeExplainer(rf)
sv_raw = explainer.shap_values(X_test_np)
sv = np.array(sv_raw)
if sv.ndim == 3: sv = sv[:,:,0]
elif isinstance(sv_raw, list): sv = np.array(sv_raw[0])
base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)): base_val = np.array(base_val).flatten()[0]

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Grafico 1: Importancia media (barras em portugues)
mean_abs_shap = np.abs(sv).mean(axis=0)
feat_imp = pd.Series(mean_abs_shap, index=feat_list).sort_values(ascending=True).tail(15)
feat_imp.plot(kind='barh', ax=axes[0], color='#e74c3c', edgecolor='black', alpha=0.8)
axes[0].set_title('Importancia Media das Features (SHAP)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Valor SHAP Medio (quanto influencia a predicao)')

# Grafico 2: Impacto por amostra (dot plot em portugues)
top15_idx = np.argsort(mean_abs_shap)[-15:]
for rank, fi in enumerate(top15_idx):
    shap_col = sv[:, fi]
    feat_col = X_test_np[:, fi]
    vmin, vmax = feat_col.min(), feat_col.max()
    colors_norm = (feat_col - vmin) / (vmax - vmin + 1e-8)
    y_jitter = rank + np.random.uniform(-0.3, 0.3, len(shap_col))
    axes[1].scatter(shap_col, y_jitter, c=colors_norm, cmap='RdBu', s=8, alpha=0.5, edgecolors='none')

axes[1].set_yticks(range(len(top15_idx)))
axes[1].set_yticklabels([feat_list[i] for i in top15_idx], fontsize=9)
axes[1].set_xlabel('Valor SHAP (positivo = empurra para maligno)')
axes[1].set_title('Impacto de Cada Feature por Paciente', fontsize=14, fontweight='bold')
axes[1].axvline(0, color='black', lw=0.8, ls='--')
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sm = ScalarMappable(cmap='RdBu', norm=Normalize(0, 1))
cbar = plt.colorbar(sm, ax=axes[1], shrink=0.5, pad=0.02)
cbar.set_label('Valor da Feature\\n(azul=baixo, vermelho=alto)', fontsize=10)
plt.tight_layout(); plt.show()
""".split('\n')
nb['cells'][21]['source'] = [line + '\n' for line in nb['cells'][21]['source']]

# CELL 27: Grad-CAM com imagens consistentes (escala de cinza)
nb['cells'][27]['source'] = """def gradcam(img_arr, modelo, layer='Conv_1'):
    lc = modelo.get_layer(layer)
    gm = tf.keras.models.Model(inputs=modelo.input, outputs=[lc.output, modelo.output])
    with tf.GradientTape() as t: co, pr = gm(img_arr); loss = pr[:,0]
    g = t.gradient(loss, co); w = tf.reduce_mean(g, axis=(0,1,2))
    h = tf.squeeze(co[0] @ w[..., tf.newaxis])
    return (tf.maximum(h, 0) / (tf.math.reduce_max(h) + 1e-8)).numpy()

real_dir = '../data/images/real_samples'; real_imgs = {}
if os.path.exists(real_dir):
    for f in sorted(os.listdir(real_dir)):
        if f.endswith(('.jpg','.png')): real_imgs[f] = cv2.imread(os.path.join(real_dir, f))
if real_imgs:
    n = len(real_imgs); fig, axes = plt.subplots(n, 3, figsize=(16, 5*n))
    if n == 1: axes = axes.reshape(1,-1)
    for idx, (nome, ibgr) in enumerate(real_imgs.items()):
        irgb = cv2.cvtColor(ibgr, cv2.COLOR_BGR2RGB)
        igray = cv2.cvtColor(ibgr, cv2.COLOR_BGR2GRAY)
        i224_gray = cv2.resize(igray, (224,224))
        i224_rgb = cv2.resize(irgb, (224,224)).astype(np.float32)/255.0
        batch = np.expand_dims(i224_rgb, 0)
        prob = model_cnn.predict(batch, verbose=0)[0][0]
        pred_lbl = 'MALIGNO' if prob > 0.5 else 'BENIGNO'
        conf = prob if prob > 0.5 else 1 - prob
        hm = gradcam(batch, model_cnn); hm_r = cv2.resize(hm, (224,224))
        i224_g3 = np.stack([i224_gray.astype(np.float32)/255.0]*3, axis=-1)
        hm_c = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*hm_r), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)/255.0
        overlay = np.clip(hm_c * 0.5 + i224_g3 * 0.7, 0, 1)
        lbl = nome.replace('mamografia_','').replace('.jpg','').replace('_',' ').title()
        axes[idx,0].imshow(i224_gray, cmap='gray')
        axes[idx,0].set_title(f'Mamografia Real\\n({lbl})', fontsize=12, fontweight='bold'); axes[idx,0].axis('off')
        axes[idx,1].imshow(hm_r, cmap='jet')
        axes[idx,1].set_title('Mapa de Calor (Grad-CAM)\\nRegioes de Atencao da IA', fontsize=12); axes[idx,1].axis('off')
        axes[idx,2].imshow(overlay)
        mask = hm_r > 0.7; coords = np.where(mask)
        if len(coords[0]) > 0:
            from matplotlib.patches import Rectangle
            axes[idx,2].add_patch(Rectangle((coords[1].min(),coords[0].min()), coords[1].max()-coords[1].min(), coords[0].max()-coords[0].min(), lw=2, ec='yellow', fc='none', ls='--'))
            axes[idx,2].text(coords[1].min(), coords[0].min()-5, 'REGIAO SUSPEITA', color='yellow', fontsize=9, fontweight='bold')
        cor = '#e74c3c' if pred_lbl=='MALIGNO' else '#2ecc71'
        axes[idx,2].set_title(f'Analise IA: {pred_lbl} ({conf:.0%})', fontsize=12, fontweight='bold', color=cor); axes[idx,2].axis('off')
    plt.suptitle('Analise de Mamografias Reais com Inteligencia Artificial', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.show()
    print('Vermelho/Amarelo = alta atencao (possivel nodulo) | Azul = baixa atencao | Retangulo = regiao suspeita')
""".split('\n')
nb['cells'][27]['source'] = [line + '\n' for line in nb['cells'][27]['source']]

# CELL 29: Caso clinico - escala de risco corrigida
nb['cells'][29]['source'] = """caso = df[df['diagnostico']==0].iloc[15]
X_caso = caso[cancer.feature_names].values.reshape(1,-1)
X_caso_s = scaler.transform(X_caso)
pred_caso = melhor_modelo.predict(X_caso_s)[0]
probs_caso = melhor_modelo.predict_proba(X_caso_s)[0]
classe_caso = 'MALIGNO' if pred_caso == 0 else 'BENIGNO'
sv_caso_raw = explainer.shap_values(X_caso_s)
sv_caso = np.array(sv_caso_raw)
if sv_caso.ndim == 3: sv_caso = sv_caso[:,:,0]
elif isinstance(sv_caso_raw, list): sv_caso = np.array(sv_caso_raw[0])

fig = plt.figure(figsize=(22, 12))

# 1. Probabilidades
ax1 = fig.add_subplot(2, 2, 1)
bars = ax1.bar(['Maligno','Benigno'], probs_caso, color=['#e74c3c','#2ecc71'], edgecolor='black', alpha=0.85, width=0.5)
for b, v in zip(bars, probs_caso): ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.1%}', ha='center', fontweight='bold', fontsize=16)
ax1.set_title(f'Resultado: {classe_caso}', fontsize=16, fontweight='bold', color='#e74c3c' if pred_caso==0 else '#2ecc71')
ax1.set_ylim(0, 1.15); ax1.set_ylabel('Probabilidade', fontsize=12)
ax1.axhline(0.5, color='gray', ls='--', alpha=0.5, label='Limiar 50%'); ax1.legend()

# 2. Escala de risco (corrigida e legivel)
ax2 = fig.add_subplot(2, 2, 2)
faixas = [(0, 0.2, '#2ecc71', 'BAIXO\\nRotina'), (0.2, 0.5, '#f1c40f', 'MODERADO\\nExames'), (0.5, 0.8, '#e67e22', 'ALTO\\nBiopsia'), (0.8, 1.0, '#e74c3c', 'MUITO ALTO\\nUrgente')]
for ini, fim, cor, txt in faixas:
    ax2.barh(0, fim-ini, left=ini, height=0.6, color=cor, edgecolor='black', alpha=0.75)
    ax2.text((ini+fim)/2, 0, txt, ha='center', va='center', fontsize=9, fontweight='bold')
pm = probs_caso[0]
ax2.plot(pm, 0.45, 'v', color='black', ms=18, zorder=5)
ax2.plot(pm, -0.45, '^', color='black', ms=18, zorder=5)
ax2.axvline(x=pm, color='black', lw=2, zorder=4)
ax2.text(pm, 0.55, f'Paciente: {pm:.0%}', ha='center', fontsize=13, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
ax2.set_xlim(0, 1); ax2.set_ylim(-0.7, 0.8); ax2.set_yticks([])
ax2.set_title('Escala de Risco do Paciente', fontsize=14, fontweight='bold')
ax2.set_xlabel('Probabilidade de Malignidade', fontsize=11)

# 3. SHAP individual (parte inferior, largura total)
ax3 = fig.add_subplot(2, 1, 2)
top_idx = np.argsort(np.abs(sv_caso[0]))[-12:]
feats = [cancer.feature_names[i] for i in top_idx]
vals = [sv_caso[0][i] for i in top_idx]
cores_shap = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
barras = ax3.barh(feats, vals, color=cores_shap, edgecolor='black', alpha=0.8)
ax3.set_title('SHAP Individual: O Que Levou a Este Diagnostico?', fontsize=14, fontweight='bold')
ax3.axvline(0, color='black', lw=1)
ax3.set_xlabel('Contribuicao (vermelho = empurra para MALIGNO | azul = empurra para BENIGNO)', fontsize=10)
for bar, val in zip(barras, vals):
    x_pos = bar.get_width() + 0.002 if val > 0 else bar.get_width() - 0.002
    ha = 'left' if val > 0 else 'right'
    ax3.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.3f}', ha=ha, va='center', fontsize=9, fontweight='bold')
plt.tight_layout(); plt.show()

if pm>0.8: risco,rec = 'MUITO ALTO','Biopsia URGENTE recomendada'
elif pm>0.5: risco,rec = 'ALTO','Biopsia recomendada + exames complementares'
elif pm>0.2: risco,rec = 'MODERADO','Acompanhamento em 6 meses + exames'
else: risco,rec = 'BAIXO','Acompanhamento de rotina'
print(f'\\nDIAGNOSTICO: {classe_caso}')
print(f'Confianca: {max(probs_caso):.1%}')
print(f'Nivel de Risco: {risco}')
print(f'Recomendacao: {rec}')
print(f'\\nATENCAO: Resultado de APOIO. O medico tem a palavra final.')
""".split('\n')
nb['cells'][29]['source'] = [line + '\n' for line in nb['cells'][29]['source']]

# Adicionar Trabalhos Futuros antes da ultima celula
trabalhos = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# PARTE 8: TRABALHOS FUTUROS\n",
        "\n",
        "### 1. Ampliacao do Dataset\n",
        "- **Problema:** 569 amostras e insuficiente para generalizacao\n",
        "- **Solucao:** Coletar dados reais + aplicar SMOTE para balancear classes\n",
        "- **Impacto:** Reducao de vies e melhoria na sensibilidade\n",
        "\n",
        "### 2. Integracao com Prontuario Eletronico\n",
        "- **Problema:** Modelo ignora historico clinico, idade, genetica\n",
        "- **Solucao:** Integrar via HL7/FHIR com idade, historico familiar, BRCA1/BRCA2\n",
        "- **Impacto:** Predicoes contextualizadas por paciente\n",
        "\n",
        "### 3. CNN com Mamografias Reais\n",
        "- **Problema:** CNN treinada em dados sinteticos\n",
        "- **Solucao:** Retreinar com CBIS-DDSM (~2.500 mamografias reais) + fine-tuning\n",
        "- **Tecnicas:** Ensemble de modelos, U-Net para segmentacao de lesoes\n",
        "- **Meta:** AUC > 0.95 em dados reais\n",
        "\n",
        "### 4. Classificacao BI-RADS\n",
        "- **Problema:** Apenas benigno vs maligno\n",
        "- **Solucao:** Expandir para escala BI-RADS (0 a 5), padrao em radiologia\n",
        "- **Impacto:** Saida alinhada com a pratica clinica real\n",
        "\n",
        "### 5. Deploy em Producao\n",
        "- API REST com FastAPI\n",
        "- Interface web com Streamlit para uso pelo medico\n",
        "- Monitoramento de drift e logging de predicoes\n",
        "\n",
        "### 6. Validacao Clinica\n",
        "- Estudo retrospectivo com casos historicos do hospital\n",
        "- Estudo prospectivo em paralelo com radiologistas\n",
        "- Submissao ao comite de etica\n",
        "\n",
        "---"
    ]
}
nb['cells'].insert(-1, trabalhos)

with open('notebooks/tech_challenge_fase1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Notebook atualizado com sucesso!')
