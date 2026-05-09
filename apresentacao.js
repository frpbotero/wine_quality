// Wine Quality Prediction — Apresentação v3
// Requer: npm install pptxgenjs react react-dom react-icons sharp
// Rodar: node wine_pptx_v3.js

const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");

const C = {
  maroon:     "6B0F1A",
  maroonDark: "3D0009",
  gold:       "B8963E",
  cream:      "FAF8F5",
  white:      "FFFFFF",
  text:       "2C1810",
  textLight:  "5C3D2E",
  green:      "1A6B2A",
};

const {
  FaBullseye, FaExclamationTriangle, FaLightbulb,
  FaCodeBranch, FaDatabase, FaChartLine, FaCog,
  FaServer, FaDesktop, FaFlask, FaSync,
  FaExpandArrowsAlt, FaEye, FaCheckCircle,
  FaCloud, FaDocker, FaTrophy,
} = require("react-icons/fa");
const { MdStorage } = require("react-icons/md");

async function iconPng(IconComp, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComp, { color, size: String(size) })
  );
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ─── helpers ──────────────────────────────────────────────────────────────────

function addSlide(pres) {
  const s = pres.addSlide();
  s.background = { color: C.cream };
  return s;
}

function addTitle(s, text, fontSize = 30) {
  s.addShape("rect", {
    x: 0.4, y: 0.22, w: 0.06, h: 0.55,
    fill: { color: C.gold }, line: { color: C.gold },
  });
  s.addText(text, {
    x: 0.58, y: 0.22, w: 9.2, h: 0.6,
    fontSize, bold: true, fontFace: "Georgia",
    color: C.maroon, valign: "middle", margin: 0,
  });
}

function makeShadow() {
  return { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.08 };
}

// ─── main ─────────────────────────────────────────────────────────────────────

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 × 5.625

  // ── SLIDE 1: Capa ────────────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.maroonDark };

    // left panel
    s.addShape("rect", {
      x: 0, y: 0, w: 4.8, h: 5.625,
      fill: { color: "1A0005" }, line: { color: "1A0005" },
    });
    s.addText("🍷", {
      x: 0.5, y: 1.2, w: 3.8, h: 3.2,
      fontSize: 90, align: "center", valign: "middle",
    });

    // right content
    s.addText("Wine Quality\nClassifier", {
      x: 5.0, y: 0.9, w: 4.7, h: 1.9,
      fontSize: 38, bold: true, fontFace: "Georgia",
      color: C.white, valign: "middle",
    });
    s.addShape("rect", {
      x: 5.0, y: 2.9, w: 1.4, h: 0.04,
      fill: { color: C.gold }, line: { color: C.gold },
    });
    s.addText("Classificação Inteligente de Vinhos\ncom Machine Learning e MLOps", {
      x: 5.0, y: 3.05, w: 4.7, h: 0.85,
      fontSize: 14, bold: true, fontFace: "Georgia", color: C.gold,
    });
    s.addText("Felipe Botero · Hanna Souza · José Henrique", {
      x: 5.0, y: 4.05, w: 4.7, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: "CCBBAA",
    });
    s.addText("Equipe de ML | Maio 2026", {
      x: 5.0, y: 4.45, w: 4.7, h: 0.3,
      fontSize: 10, fontFace: "Calibri", color: "998877",
    });
  }

  // ── SLIDE 2: Visão Geral ──────────────────────────────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "Visão Geral do Projeto");

    const cards = [
      {
        icon: FaBullseye,
        label: "Objetivo",
        text: "Classificar vinhos em \"Bom\" ou \"Ruim\" com acurácia >85%, usando 11 atributos físico-químicos e o tipo do vinho.",
      },
      {
        icon: FaExclamationTriangle,
        label: "O Desafio",
        text: "Desbalanceamento crítico: ~93% dos dados nas notas 5–7. Classificação multiclasse original limitada a F1 ≈ 62–63%.",
      },
      {
        icon: FaLightbulb,
        label: "A Solução",
        text: "Binarização com threshold 7 + SMOTE e TomekLinks + 6 algoritmos = 12 modelos comparados automaticamente via MLflow.",
      },
    ];

    let y = 1.05;
    for (const card of cards) {
      const ico = await iconPng(card.icon, "#" + C.maroon);
      s.addShape("rect", {
        x: 0.4, y, w: 4.5, h: 1.1,
        fill: { color: C.white }, line: { color: "E0D4CC", width: 0.5 },
        shadow: makeShadow(),
      });
      s.addImage({ data: ico, x: 0.55, y: y + 0.3, w: 0.36, h: 0.36 });
      s.addText(card.label, {
        x: 1.05, y: y + 0.06, w: 3.7, h: 0.32,
        fontSize: 13, bold: true, fontFace: "Georgia", color: C.maroon, margin: 0,
      });
      s.addText(card.text, {
        x: 1.05, y: y + 0.38, w: 3.7, h: 0.65,
        fontSize: 10.5, fontFace: "Calibri", color: C.text, margin: 0,
      });
      y += 1.25;
    }

    // right KPIs
    const kpis = [
      { val: "95.08%", sub: "Acurácia · Teste (hold-out)" },
      { val: "95.03%", sub: "F1 Teste  |  87.43% F1 Val" },
      { val: "Tomek+RF", sub: "Modelo Campeão (@champion)" },
    ];
    let ky = 1.05;
    for (const kpi of kpis) {
      s.addShape("rect", {
        x: 5.3, y: ky, w: 4.3, h: 1.1,
        fill: { color: C.maroon }, line: { color: C.maroon },
      });
      s.addText(kpi.val, {
        x: 5.3, y: ky + 0.06, w: 4.3, h: 0.58,
        fontSize: 28, bold: true, fontFace: "Georgia",
        color: C.gold, align: "center", margin: 0,
      });
      s.addText(kpi.sub, {
        x: 5.3, y: ky + 0.65, w: 4.3, h: 0.35,
        fontSize: 11, fontFace: "Calibri",
        color: "EEE0D0", align: "center", margin: 0,
      });
      ky += 1.25;
    }
  }

  // ── SLIDE 3: EDA e Engenharia de Atributos ───────────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "EDA e Engenharia de Atributos");

    const insights = [
      {
        head: "Dataset",
        body: "6.497 amostras: ~1.599 tintos + ~4.898 brancos. 11 atributos físico-químicos + tipo do vinho.",
      },
      {
        head: "Desbalanceamento Crítico",
        body: "Notas 5, 6 e 7 = ~93% dos dados. Binarização (≥7 → Bom) gerou +33pp de F1 imediato.",
      },
      {
        head: "Variabilidade por Tipo",
        body: "Brancos: maior açúcar residual e SO₂. Tintos: maior acidez volátil e sulfatos → One-Hot Encoding obrigatório.",
      },
      {
        head: "Pipeline de Features",
        body: "13 features totais: 11 físico-químicas + type_red + type_white. StandardScaler fitado só no treino.",
      },
    ];

    let iy = 1.0;
    for (const ins of insights) {
      s.addShape("rect", {
        x: 0.4, y: iy, w: 0.06, h: 0.88,
        fill: { color: C.gold }, line: { color: C.gold },
      });
      s.addText(ins.head, {
        x: 0.58, y: iy, w: 4.6, h: 0.32,
        fontSize: 12, bold: true, fontFace: "Georgia", color: C.maroon, margin: 0,
      });
      s.addText(ins.body, {
        x: 0.58, y: iy + 0.32, w: 4.6, h: 0.56,
        fontSize: 10.5, fontFace: "Calibri", color: C.text, margin: 0,
      });
      iy += 1.08;
    }

    // right: custom correlation bar chart (manual — pptxgenjs rounds decimals)
    const chartX = 5.4;
    const chartY = 1.0;
    const chartW = 4.2;
    const chartH = 4.3;

    s.addShape("rect", {
      x: chartX, y: chartY, w: chartW, h: chartH,
      fill: { color: C.cream }, line: { color: "E0D4CC", width: 0.5 },
    });
    s.addText("Correlação com Qualidade", {
      x: chartX, y: chartY + 0.1, w: chartW, h: 0.3,
      fontSize: 11, bold: true, fontFace: "Georgia",
      color: C.maroon, align: "center", margin: 0,
    });

    const corrData = [
      { label: "Álcool",        val:  0.44, color: C.maroon },
      { label: "Sulfatos",      val:  0.25, color: C.maroon },
      { label: "Ácido Cítrico", val:  0.09, color: C.maroon },
      { label: "Cloretos",      val: -0.04, color: "AAAAAA" },
      { label: "Dióx.Enxofre", val: -0.18, color: "AAAAAA" },
      { label: "Densidade",     val: -0.31, color: "A06040" },
      { label: "Acidez Vol.",   val: -0.27, color: "A06040" },
    ];

    const labelW  = 1.1;
    const barAreaX = chartX + labelW;
    const barAreaW = chartW - labelW - 0.1;
    const halfW    = barAreaW / 2;
    const zeroX    = barAreaX + halfW;
    const maxVal   = 0.5;
    const rowsY    = chartY + 0.52;
    const rowH     = (chartH - 0.58) / corrData.length;

    // zero line
    s.addShape("rect", {
      x: zeroX, y: rowsY - 0.05, w: 0.008, h: corrData.length * rowH + 0.05,
      fill: { color: "888888" }, line: { color: "888888" },
    });

    for (let i = 0; i < corrData.length; i++) {
      const { label, val, color } = corrData[i];
      const ry     = rowsY + i * rowH;
      const barH   = rowH * 0.48;
      const barY   = ry + (rowH - barH) / 2;
      const barLen = Math.abs(val) / maxVal * halfW;
      const barX   = val >= 0 ? zeroX : zeroX - barLen;
      const valStr = (val > 0 ? "+" : "") + val.toFixed(2);

      // category label
      s.addText(label, {
        x: chartX + 0.05, y: ry, w: labelW - 0.1, h: rowH,
        fontSize: 9, fontFace: "Calibri", color: C.textLight,
        align: "right", valign: "middle", margin: 0,
      });

      // bar
      if (barLen > 0.01) {
        s.addShape("rect", {
          x: barX, y: barY, w: barLen, h: barH,
          fill: { color }, line: { color },
        });
      }

      // value label beside bar
      if (val >= 0) {
        s.addText(valStr, {
          x: barX + barLen + 0.04, y: ry, w: 0.42, h: rowH,
          fontSize: 8.5, bold: true, fontFace: "Calibri", color: C.text,
          align: "left", valign: "middle", margin: 0,
        });
      } else {
        s.addText(valStr, {
          x: barX - 0.46, y: ry, w: 0.42, h: rowH,
          fontSize: 8.5, bold: true, fontFace: "Calibri", color: C.text,
          align: "right", valign: "middle", margin: 0,
        });
      }
    }
  }

  // ── SLIDE 4: Estratégia de Classificação Binária ──────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "Estratégia de Classificação Binária");

    // ANTES box
    s.addShape("rect", {
      x: 0.4, y: 1.0, w: 4.1, h: 4.1,
      fill: { color: "F0E8E0" }, line: { color: "D4B8A0", width: 1 },
    });
    s.addShape("rect", {
      x: 0.4, y: 1.0, w: 4.1, h: 0.5,
      fill: { color: "A06040" }, line: { color: "A06040" },
    });
    s.addText("ANTES — Multiclasse (3–9)", {
      x: 0.4, y: 1.0, w: 4.1, h: 0.5,
      fontSize: 13, bold: true, fontFace: "Georgia",
      color: C.white, align: "center", valign: "middle", margin: 0,
    });

    const before = [
      "Notas 3 a 9 como classes distintas",
      "~93% dos dados nas notas 5, 6 e 7",
      "Nota 4: apenas 49 amostras no teste",
      "F1 de apenas 32% nas classes raras",
      "F1 médio ≈ 62–63% (melhor modelo)",
    ];
    let by = 1.62;
    for (const t of before) {
      s.addText([
        { text: "✗  ", options: { color: "CC4444", bold: true } },
        { text: t, options: { color: C.text } },
      ], { x: 0.6, y: by, w: 3.7, h: 0.44, fontSize: 11, fontFace: "Calibri", margin: 0 });
      by += 0.48;
    }

    // arrow
    s.addText("→", {
      x: 4.55, y: 2.7, w: 0.8, h: 0.7,
      fontSize: 32, color: C.gold, bold: true, align: "center", margin: 0,
    });

    // AGORA box
    s.addShape("rect", {
      x: 5.45, y: 1.0, w: 4.15, h: 4.1,
      fill: { color: "F5F0EA" }, line: { color: C.gold, width: 1.5 },
    });
    s.addShape("rect", {
      x: 5.45, y: 1.0, w: 4.15, h: 0.5,
      fill: { color: C.maroon }, line: { color: C.maroon },
    });
    s.addText("AGORA — Binário (threshold = 7)", {
      x: 5.45, y: 1.0, w: 4.15, h: 0.5,
      fontSize: 13, bold: true, fontFace: "Georgia",
      color: C.gold, align: "center", valign: "middle", margin: 0,
    });

    const after = [
      "0 = Ruim (nota < 7)  |  1 = Bom (nota ≥ 7)",
      "Classes com volume suficiente para aprendizado",
      "Fronteiras de decisão limpas e coesas",
      "SQL: CASE WHEN quality < 7 THEN 0 ELSE 1 END",
      "F1 de 95.03% com Tomek + Random Forest",
    ];
    let ay = 1.62;
    for (const t of after) {
      s.addText([
        { text: "✓  ", options: { color: C.green, bold: true } },
        { text: t, options: { color: C.text } },
      ], { x: 5.65, y: ay, w: 3.75, h: 0.44, fontSize: 11, fontFace: "Calibri", margin: 0 });
      ay += 0.48;
    }

    // bottom gain tag
    s.addShape("rect", {
      x: 0.4, y: 5.18, w: 9.2, h: 0.32,
      fill: { color: C.maroon }, line: { color: C.maroon },
    });
    s.addText("Ganho real: +33 pontos percentuais de F1 (de ~62% multiclasse → 95% binário)", {
      x: 0.4, y: 5.18, w: 9.2, h: 0.32,
      fontSize: 11, fontFace: "Calibri", bold: true,
      color: C.gold, align: "center", valign: "middle", margin: 0,
    });
  }

  // ── SLIDE 5: Stack Tecnológico ────────────────────────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "Stack Tecnológico");

    const stack = [
      { icon: FaDatabase,  label: "Dados",       tech: "DuckDB + SQL\n+ Supabase Postgres",  cat: "Ingestão & Features" },
      { icon: FaCodeBranch,label: "Pipeline",    tech: "DVC",                                 cat: "Versionamento" },
      { icon: FaChartLine, label: "Tracking",    tech: "MLflow\n+ DagsHub",                  cat: "Experimentos" },
      { icon: FaCog,       label: "Modelagem",   tech: "scikit-learn\n+ XGBoost",             cat: "6 Algoritmos · 12 Modelos" },
      { icon: FaFlask,     label: "Balanc.",     tech: "SMOTE\n+ TomekLinks",                 cat: "imbalanced-learn" },
      { icon: FaDesktop,   label: "Interface",   tech: "Streamlit\n(3 abas)",                 cat: "Predição · Comparação · Histórico" },
      { icon: FaCog,       label: "Deploy",      tech: "Docker\n+ Render",                    cat: "Infra & Hospedagem" },
    ];

    // 4 top + 3 bottom
    for (let i = 0; i < stack.length; i++) {
      const item = stack[i];
      const ico = await iconPng(item.icon, "#" + C.gold);
      const bx = i < 4 ? 0.4 + i * 2.4 : 0.4 + (i - 4) * 3.1;
      const by = i < 4 ? 1.0 : 3.1;
      const bw = i < 4 ? 2.15 : 2.85;

      s.addShape("rect", {
        x: bx, y: by, w: bw, h: 0.45,
        fill: { color: C.maroon }, line: { color: C.maroon },
      });
      s.addImage({ data: ico, x: bx + bw / 2 - 0.15, y: by + 0.07, w: 0.28, h: 0.28 });

      s.addShape("rect", {
        x: bx, y: by + 0.45, w: bw, h: 1.55,
        fill: { color: C.white }, line: { color: "E0D4CC", width: 0.5 },
        shadow: makeShadow(),
      });
      s.addText(item.label, {
        x: bx, y: by + 0.5, w: bw, h: 0.32,
        fontSize: 10, bold: true, fontFace: "Georgia",
        color: C.maroon, align: "center", margin: 0,
      });
      s.addText(item.tech, {
        x: bx, y: by + 0.8, w: bw, h: 0.65,
        fontSize: 11, fontFace: "Calibri",
        color: C.textLight, align: "center", margin: 0,
      });
      s.addText(item.cat, {
        x: bx, y: by + 1.45, w: bw, h: 0.5,
        fontSize: 8.5, fontFace: "Calibri",
        color: C.gold, align: "center", italic: true, margin: 0,
      });
    }
  }

  // ── SLIDE 6: Pipeline de Dados (DVC) ─────────────────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "Pipeline de Dados (DVC)");

    const stages = [
      { name: "INGESTION",    desc: "Coleta do Supabase (Postgres) ou Kaggle (fallback). Normaliza colunas → data/raw/" },
      { name: "PREPROCESSING",desc: "DuckDB binariza o target via SQL. Salva data/processed/wine_processed.parquet" },
      { name: "PREPARE DATA", desc: "Split estratificado 60/20/20 (random_state=42). Gera train, val e test parquets" },
      { name: "TRAIN",        desc: "12 modelos treinados. Registra no MLflow. Campeão promovido como @champion" },
      { name: "EVALUATE",     desc: "Todos os 12 avaliados no test set puro. Gera evaluation_report.json e curvas ROC" },
    ];

    // left: pipeline steps
    let sy = 1.05;
    for (let i = 0; i < stages.length; i++) {
      const stage = stages[i];
      s.addShape("rect", {
        x: 0.4, y: sy, w: 4.5, h: 0.72,
        fill: { color: i === 3 ? C.maroon : "F0E8E0" },
        line: { color: i === 3 ? C.maroon : C.gold, width: 1 },
      });
      s.addText(stage.name, {
        x: 0.5, y: sy + 0.04, w: 1.3, h: 0.3,
        fontSize: 10, bold: true, fontFace: "Georgia",
        color: i === 3 ? C.gold : C.maroon, margin: 0,
      });
      s.addText(stage.desc, {
        x: 0.5, y: sy + 0.34, w: 4.2, h: 0.33,
        fontSize: 9.5, fontFace: "Calibri",
        color: i === 3 ? "EEE0D0" : C.text, margin: 0,
      });
      if (i < stages.length - 1) {
        s.addText("↓", {
          x: 0.4, y: sy + 0.74, w: 4.5, h: 0.2,
          fontSize: 13, color: C.gold, align: "center", margin: 0,
        });
        sy += 0.95;
      }
    }

    // right: benefits
    s.addShape("rect", {
      x: 5.25, y: 1.05, w: 4.35, h: 4.3,
      fill: { color: C.maroon }, line: { color: C.maroon },
    });
    s.addText("Benefícios do DVC", {
      x: 5.25, y: 1.1, w: 4.35, h: 0.42,
      fontSize: 14, bold: true, fontFace: "Georgia",
      color: C.gold, align: "center", margin: 0,
    });
    s.addShape("rect", {
      x: 5.45, y: 1.58, w: 3.95, h: 0.03,
      fill: { color: C.gold }, line: { color: C.gold },
    });

    const benefits = [
      { title: "Reprodutibilidade Total", body: "dvc repro reconstrói o pipeline do zero detectando mudanças por hash dos inputs." },
      { title: "Versionamento de Dados", body: "Parquets e modelos rastreados — qualquer estado anterior pode ser restaurado." },
      { title: "MLflow + DagsHub", body: "Tracking remoto com colaboração distribuída quando credenciais configuradas." },
      { title: "Fallback Inteligente", body: "Sem DagsHub → MLflow local. Sem MLflow → JSON estático em reports/." },
    ];
    let br = 1.7;
    for (const b of benefits) {
      s.addText(b.title, {
        x: 5.4, y: br, w: 4.1, h: 0.3,
        fontSize: 11, bold: true, fontFace: "Georgia",
        color: C.gold, margin: 0,
      });
      s.addText(b.body, {
        x: 5.4, y: br + 0.28, w: 4.1, h: 0.38,
        fontSize: 10, fontFace: "Calibri", color: "EEE0D0", margin: 0,
      });
      br += 0.82;
    }
  }

  // ── SLIDE 7: Treinamento — 12 Modelos ────────────────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "Treinamento e Seleção de Modelos");

    const methods = [
      {
        icon: FaCog,
        title: "6 Algoritmos Avaliados",
        body: "Logistic Regression, Random Forest, KNN, SVM, XGBoost e MLP — cobrindo abordagens lineares, ensemble e redes neurais.",
      },
      {
        icon: FaFlask,
        title: "2 Estratégias de Balanceamento",
        body: "SMOTE (oversampling sintético) e TomekLinks (undersampling por limpeza de fronteira) → 12 combinações treinadas.",
      },
      {
        icon: FaChartLine,
        title: "Métrica de Seleção",
        body: "F1-Score Ponderado como critério principal, registrado por run no MLflow. Campeão recebe alias @champion no Model Registry.",
      },
    ];

    let my = 1.0;
    for (const m of methods) {
      const ico = await iconPng(m.icon, "#" + C.maroon);
      s.addShape("rect", {
        x: 0.4, y: my, w: 4.6, h: 1.22,
        fill: { color: C.white }, line: { color: "E0D4CC", width: 0.5 },
        shadow: makeShadow(),
      });
      s.addImage({ data: ico, x: 0.55, y: my + 0.35, w: 0.36, h: 0.36 });
      s.addText(m.title, {
        x: 1.05, y: my + 0.08, w: 3.8, h: 0.36,
        fontSize: 13, bold: true, fontFace: "Georgia", color: C.maroon, margin: 0,
      });
      s.addText(m.body, {
        x: 1.05, y: my + 0.44, w: 3.8, h: 0.72,
        fontSize: 10.5, fontFace: "Calibri", color: C.text, margin: 0,
      });
      my += 1.42;
    }

    // right: metrics panel
    s.addShape("rect", {
      x: 5.3, y: 1.0, w: 4.3, h: 4.3,
      fill: { color: C.maroon }, line: { color: C.maroon },
    });
    s.addText("Split & Métricas", {
      x: 5.3, y: 1.06, w: 4.3, h: 0.45,
      fontSize: 15, bold: true, fontFace: "Georgia",
      color: C.gold, align: "center", margin: 0,
    });
    s.addShape("rect", {
      x: 5.5, y: 1.57, w: 3.9, h: 0.03,
      fill: { color: C.gold }, line: { color: C.gold },
    });

    const infos = [
      { label: "Split treino/val/teste", val: "60 / 20 / 20" },
      { label: "Split estratificado", val: "random_state=42" },
      { label: "Val F1 (seleção)", val: "87.43% ✓" },
      { label: "Test F1 (resultado)", val: "95.03% ✓" },
      { label: "Total de runs MLflow", val: "12 modelos" },
    ];
    let ry = 1.7;
    for (const info of infos) {
      s.addText(info.label, {
        x: 5.5, y: ry, w: 2.4, h: 0.4,
        fontSize: 11.5, fontFace: "Calibri", color: C.white, margin: 0,
      });
      s.addText(info.val, {
        x: 7.8, y: ry, w: 1.7, h: 0.4,
        fontSize: 11.5, fontFace: "Calibri",
        color: C.gold, align: "right", bold: true, margin: 0,
      });
      ry += 0.58;
    }
  }

  // ── SLIDE 8: Comparação Real Multiclasse vs Binário ───────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "Multiclasse vs Binário — Comparativo Real", 26);

    // note
    s.addText("Mesmos algoritmos, dados e estratégias de balanceamento — única diferença é a definição do target.", {
      x: 0.4, y: 0.88, w: 9.2, h: 0.3,
      fontSize: 10, fontFace: "Calibri", italic: true, color: C.textLight, margin: 0,
    });

    // table
    const rows = [
      // header
      [
        { text: "Modelo", options: { bold: true, color: C.gold, fill: { color: C.maroon }, align: "left" } },
        { text: "Multiclasse F1", options: { bold: true, color: C.gold, fill: { color: C.maroon }, align: "center" } },
        { text: "Binário F1", options: { bold: true, color: C.gold, fill: { color: C.maroon }, align: "center" } },
        { text: "Ganho", options: { bold: true, color: C.gold, fill: { color: C.maroon }, align: "center" } },
      ],
      [
        { text: "★ Tomek_random_forest", options: { color: C.maroon, bold: true } },
        { text: "62.30%", options: { color: C.textLight, align: "center" } },
        { text: "95.03%", options: { color: C.green, bold: true, align: "center" } },
        { text: "+32.73pp", options: { color: C.green, bold: true, align: "center" } },
      ],
      [
        { text: "Tomek_xgboost", options: { color: C.text } },
        { text: "59.71%", options: { color: C.textLight, align: "center" } },
        { text: "94.90%", options: { color: C.green, bold: true, align: "center" } },
        { text: "+35.19pp", options: { color: C.green, bold: true, align: "center" } },
      ],
      [
        { text: "SMOTE_xgboost", options: { color: C.text } },
        { text: "57.69%", options: { color: C.textLight, align: "center" } },
        { text: "94.60%", options: { color: C.green, bold: true, align: "center" } },
        { text: "+36.91pp", options: { color: C.green, bold: true, align: "center" } },
      ],
      [
        { text: "SMOTE_random_forest", options: { color: C.text } },
        { text: "58.80%", options: { color: C.textLight, align: "center" } },
        { text: "93.97%", options: { color: C.green, bold: true, align: "center" } },
        { text: "+35.17pp", options: { color: C.green, bold: true, align: "center" } },
      ],
      [
        { text: "Tomek_logistic_reg.", options: { color: C.text } },
        { text: "58.98%", options: { color: C.textLight, align: "center" } },
        { text: "79.60%", options: { color: C.green, bold: true, align: "center" } },
        { text: "+20.62pp", options: { color: C.green, bold: true, align: "center" } },
      ],
      [
        { text: "SMOTE_logistic_reg.", options: { color: C.text } },
        { text: "55.42%", options: { color: C.textLight, align: "center" } },
        { text: "75.08%", options: { color: C.green, bold: true, align: "center" } },
        { text: "+19.66pp", options: { color: C.green, bold: true, align: "center" } },
      ],
    ];

    s.addTable(rows, {
      x: 0.4, y: 1.22, w: 9.2, h: 3.5,
      border: { pt: 1, color: "E0D4CC" },
      colW: [3.2, 2.0, 2.0, 2.0],
      fontSize: 11.5,
      fontFace: "Calibri",
    });

    // bottom note
    s.addShape("rect", {
      x: 0.4, y: 4.9, w: 9.2, h: 0.55,
      fill: { color: "F0E8E0" }, line: { color: C.gold, width: 0.5 },
    });
    s.addText("Multiclasse falha nas classes raras (nota 4: F1=32%). Binário cria blocos coesos e eleva o ganho médio para +35pp.", {
      x: 0.55, y: 4.93, w: 9.0, h: 0.45,
      fontSize: 10.5, fontFace: "Calibri", italic: true, color: C.text, margin: 0,
    });
  }

  // ── SLIDE 9: Resultados no Teste ──────────────────────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "Resultados no Teste (Dados Não Vistos)");

    s.addText("A UI exibe métricas de validação (usadas na seleção do modelo). As métricas abaixo são do conjunto de TESTE — nunca visto durante treino ou validação.", {
      x: 0.4, y: 0.88, w: 9.2, h: 0.3,
      fontSize: 9.5, fontFace: "Calibri", italic: true, color: C.textLight, margin: 0,
    });

    // podium left
    const podium = [
      { medal: "🥇", val: "95.03%", label: "Tomek + Random Forest", sub: "Campeão (@champion). Alta robustez e generalização no teste puro." },
      { medal: "🥈", val: "94.90%", label: "Tomek + XGBoost", sub: "Desempenho muito próximo — captura interações não lineares." },
      { medal: "🥉", val: "94.60%", label: "SMOTE + XGBoost", sub: "Boosting com oversampling sintético — performance consistente." },
    ];

    let py = 1.0;
    for (const p of podium) {
      s.addShape("rect", {
        x: 0.4, y: py, w: 0.06, h: 1.15,
        fill: { color: C.maroon }, line: { color: C.maroon },
      });
      s.addText(p.medal + " " + p.val, {
        x: 0.6, y: py + 0.02, w: 2.4, h: 0.48,
        fontSize: 22, bold: true, fontFace: "Georgia", color: C.maroon, margin: 0,
      });
      s.addText(p.label, {
        x: 0.6, y: py + 0.48, w: 4.5, h: 0.28,
        fontSize: 12, bold: true, fontFace: "Georgia", color: C.text, margin: 0,
      });
      s.addText(p.sub, {
        x: 0.6, y: py + 0.74, w: 4.5, h: 0.34,
        fontSize: 10.5, fontFace: "Calibri", color: C.textLight, margin: 0,
      });
      py += 1.38;
    }

    // right: bar chart all models
    s.addChart(pres.charts.BAR, [
      {
        name: "Test F1 (%)",
        labels: [
          "SMOTE_logistic_reg.", "SMOTE_svm", "Tomek_logistic_reg.",
          "SMOTE_knn", "Tomek_svm", "Tomek_knn",
          "Tomek_mlp", "SMOTE_mlp", "SMOTE_random_forest",
          "SMOTE_xgboost", "Tomek_xgboost", "Tomek_random_forest",
        ],
        values: [75.08, 78.96, 79.60, 82.42, 81.49, 86.36, 93.04, 93.55, 93.97, 94.60, 94.90, 95.03],
      },
    ], {
      x: 5.35, y: 0.95, w: 4.25, h: 4.4,
      barDir: "bar",
      chartColors: [
        "BBBBBB","BBBBBB","BBBBBB","BBBBBB","BBBBBB","BBBBBB",
        "8B4A52","8B4A52","8B4A52","8B4A52","5C1A28", C.maroon,
      ],
      chartArea: { fill: { color: C.cream } },
      catAxisLabelColor: C.textLight,
      valAxisLabelColor: C.textLight,
      valGridLine: { color: "E2D8D0", size: 0.5 },
      catGridLine: { style: "none" },
      showValue: true,
      dataLabelColor: C.text,
      dataLabelFontSize: 8,
      showLegend: false,
      showTitle: true,
      title: "F1 no Teste — 12 Modelos (%)",
      titleColor: C.maroon,
      titleFontSize: 11,
    });
  }

  // ── SLIDE 10: Arquitetura de Deploy ──────────────────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "Arquitetura de Deploy");

    const tools = [
      {
        icon: FaDesktop,
        name: "Streamlit",
        desc: "3 abas: Predição (sliders + probabilidades), Comparação (val vs teste, F1 charts) e Histórico (predições salvas no Supabase).",
      },
      {
        icon: FaCog,
        name: "Docker + Render",
        desc: "Containerização completa via Docker. Deploy hospedado no Render com variáveis de ambiente injetadas via .env.",
      },
      {
        icon: FaDatabase,
        name: "Supabase",
        desc: "Fonte de verdade dos dados de treino e backbone operacional: registra cada predição da UI via supabase_logger.py.",
      },
    ];

    let tx = 0.4;
    for (const tool of tools) {
      const ico = await iconPng(tool.icon, "#" + C.white);
      s.addShape("rect", {
        x: tx, y: 1.0, w: 2.97, h: 3.8,
        fill: { color: C.white }, line: { color: "E0D4CC", width: 0.5 },
        shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.1 },
      });
      s.addShape("oval", {
        x: tx + 0.96, y: 1.2, w: 1.05, h: 1.05,
        fill: { color: C.maroon }, line: { color: C.maroon },
      });
      s.addImage({ data: ico, x: tx + 1.11, y: 1.35, w: 0.75, h: 0.75 });
      s.addText(tool.name, {
        x: tx, y: 2.38, w: 2.97, h: 0.42,
        fontSize: 15, bold: true, fontFace: "Georgia",
        color: C.maroon, align: "center", margin: 0,
      });
      s.addText(tool.desc, {
        x: tx + 0.12, y: 2.88, w: 2.73, h: 1.8,
        fontSize: 10.5, fontFace: "Calibri",
        color: C.text, align: "center", margin: 0,
      });
      tx += 3.2;
    }

    // bottom quote
    s.addShape("rect", {
      x: 0.4, y: 4.95, w: 9.2, h: 0.52,
      fill: { color: "F0E8E0" }, line: { color: C.gold, width: 0.5 },
    });
    s.addText('"Arquitetura desacoplada: Streamlit consome joblib diretamente — sem backend extra. Supabase centraliza dados e auditoria."', {
      x: 0.55, y: 4.98, w: 9.0, h: 0.44,
      fontSize: 10.5, fontFace: "Georgia", italic: true,
      color: C.textLight, align: "center", margin: 0,
    });
  }

  // ── SLIDE 11: O Que Está Sendo Feito Agora ───────────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "O Que Está Sendo Feito Agora", 28);

    const cols = [
      {
        icon: FaSync,
        title: "Operação & MLOps",
        items: [
          "Pipeline único treino/inferência via DVC",
          "Versionamento por experimento (dados + modelos)",
          "Tracking completo com MLflow + DagsHub",
        ],
      },
      {
        icon: FaChartLine,
        title: "Monitoramento de Modelo",
        items: [
          "Preparação para drift de features e predições",
          "Alertas planejados por F1, Recall e AUC",
          "Gatilho de re-treino via CI/CD (GitHub Actions)",
        ],
      },
      {
        icon: FaExpandArrowsAlt,
        title: "Expansão & Qualidade",
        items: [
          "Ampliação da base com novas safras/regiões",
          "Validação de qualidade dos dados na ingestão",
          "Supabase como fonte de verdade centralizada",
        ],
      },
    ];

    let cx = 0.4;
    for (const col of cols) {
      const ico = await iconPng(col.icon, "#" + C.white);
      s.addShape("rect", {
        x: cx, y: 1.0, w: 2.97, h: 0.55,
        fill: { color: C.maroon }, line: { color: C.maroon },
      });
      s.addImage({ data: ico, x: cx + 0.12, y: 1.08, w: 0.3, h: 0.3 });
      s.addText(col.title, {
        x: cx + 0.5, y: 1.03, w: 2.35, h: 0.5,
        fontSize: 12, bold: true, fontFace: "Georgia",
        color: C.gold, valign: "middle", margin: 0,
      });
      s.addShape("rect", {
        x: cx, y: 1.55, w: 2.97, h: 3.7,
        fill: { color: C.white }, line: { color: "E0D4CC", width: 0.5 },
        shadow: makeShadow(),
      });
      let iy = 1.72;
      for (const item of col.items) {
        s.addShape("rect", {
          x: cx + 0.18, y: iy + 0.1, w: 0.08, h: 0.08,
          fill: { color: C.gold }, line: { color: C.gold },
        });
        s.addText(item, {
          x: cx + 0.38, y: iy, w: 2.45, h: 0.55,
          fontSize: 10.5, fontFace: "Calibri", color: C.text, margin: 0,
        });
        iy += 0.65;
      }
      cx += 3.2;
    }
  }

  // ── SLIDE 12: Próximos Passos ─────────────────────────────────────────────────
  {
    const s = addSlide(pres);
    addTitle(s, "Próximos Passos");

    const steps = [
      {
        num: "01", icon: FaChartLine,
        title: "Monitoramento de Drift",
        body: "Evidently ou Alibi Detect para drift de features e predições em produção, com alertas automáticos.",
      },
      {
        num: "02", icon: FaSync,
        title: "CI/CD + Re-treino Automático",
        body: "GitHub Actions para re-treino ao detectar drift, com rollback automático via MLflow Registry.",
      },
      {
        num: "03", icon: FaExpandArrowsAlt,
        title: "Expansão do Dataset",
        body: "Novas safras e regiões para melhorar generalização e cobrir melhor os perfis extremos de qualidade.",
      },
      {
        num: "04", icon: FaEye,
        title: "Explicabilidade (SHAP)",
        body: "SHAP values integrados na UI do Streamlit para transparência e confiança na tomada de decisão.",
      },
    ];

    let py = 1.0;
    let col = 0;
    for (const step of steps) {
      const ico = await iconPng(step.icon, "#" + C.white);
      const px = col === 0 ? 0.4 : 5.3;

      s.addShape("rect", {
        x: px, y: py, w: 4.5, h: 1.88,
        fill: { color: C.white }, line: { color: "E0D4CC", width: 0.5 },
        shadow: makeShadow(),
      });
      s.addShape("rect", {
        x: px, y: py, w: 0.72, h: 1.88,
        fill: { color: C.maroon }, line: { color: C.maroon },
      });
      s.addText(step.num, {
        x: px, y: py + 0.08, w: 0.72, h: 0.5,
        fontSize: 18, bold: true, fontFace: "Georgia",
        color: C.gold, align: "center", margin: 0,
      });
      s.addImage({ data: ico, x: px + 0.17, y: py + 0.72, w: 0.38, h: 0.38 });
      s.addText(step.title, {
        x: px + 0.84, y: py + 0.1, w: 3.5, h: 0.4,
        fontSize: 13, bold: true, fontFace: "Georgia", color: C.maroon, margin: 0,
      });
      s.addText(step.body, {
        x: px + 0.84, y: py + 0.52, w: 3.5, h: 1.25,
        fontSize: 10.5, fontFace: "Calibri", color: C.text, margin: 0,
      });

      col++;
      if (col === 2) { col = 0; py += 2.1; }
    }
  }

  // ── SLIDE 13: Encerramento ────────────────────────────────────────────────────
  {
    const s = pres.addSlide();
    s.background = { color: C.maroonDark };

    s.addShape("rect", {
      x: 0, y: 0, w: 4.8, h: 5.625,
      fill: { color: "1A0005" }, line: { color: "1A0005" },
    });
    s.addText("🍷", {
      x: 0.5, y: 1.5, w: 3.8, h: 2.6,
      fontSize: 80, align: "center", valign: "middle",
    });

    s.addText("Obrigado!", {
      x: 5.0, y: 1.2, w: 4.7, h: 0.95,
      fontSize: 42, bold: true, fontFace: "Georgia",
      color: C.white, align: "center",
    });
    s.addShape("rect", {
      x: 6.2, y: 2.25, w: 1.5, h: 0.04,
      fill: { color: C.gold }, line: { color: C.gold },
    });
    s.addText("Dúvidas?", {
      x: 5.0, y: 2.42, w: 4.7, h: 0.5,
      fontSize: 19, bold: true, fontFace: "Georgia",
      color: C.gold, align: "center",
    });

    const links = [
      { icon: "📦", label: "GitHub", val: "github.com/frpbotero/wine_quality" },
      { icon: "🚀", label: "Deploy", val: "Render (Streamlit + Docker)" },
      { icon: "📊", label: "Tracking", val: "DagsHub / MLflow" },
      { icon: "🗄️", label: "Dados", val: "Supabase PostgreSQL" },
    ];
    let ly = 3.1;
    for (const link of links) {
      s.addText(link.icon + "  " + link.label + ": " + link.val, {
        x: 5.0, y: ly, w: 4.7, h: 0.34,
        fontSize: 11, fontFace: "Calibri", color: "CCBBAA", align: "center", margin: 0,
      });
      ly += 0.38;
    }

    s.addText("Felipe Botero · Hanna Souza · José Henrique", {
      x: 5.0, y: 5.1, w: 4.7, h: 0.35,
      fontSize: 10, fontFace: "Calibri", color: "887766", align: "center", margin: 0,
    });
  }

  await pres.writeFile({ fileName: "Wine_Quality_Prediction_v3.pptx" });
  console.log("✅ Wine_Quality_Prediction_v3.pptx gerado com sucesso!");
}

main().catch(console.error);
