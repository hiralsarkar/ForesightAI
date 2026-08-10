// Foresight AI - 6 slide business-context deck. Dark navy / amber, matches the live dashboard.
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const Fi = require("react-icons/fi");
const sharp = require("sharp");

// ---- palette (from app/theme.py) ----
const NAVY = "0A1628", PANEL = "0F1E33", PANEL2 = "152740", BORDER = "1E3350";
const AMBER = "F59E0B", WHITE = "FFFFFF", DIM = "94A3B8";
const GOOD = "22C55E", BAD = "EF4444", ORANGE = "F97316";
const F = "Calibri";

// ---- icon rasterizer -> base64 png ----
async function icon(name, hex, px = 300) {
  let svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Fi[name]));
  // exact-attr replaces: do NOT touch stroke-width (regex on width= would corrupt it)
  svg = svg.replace(/currentColor/g, "#" + hex)
    .replace('width="1em"', `width="${px}"`)
    .replace('height="1em"', `height="${px}"`);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const NAV = {};
  const AMB = {};
  const names = ["FiClock","FiSearch","FiRss","FiDownloadCloud","FiActivity","FiGitMerge",
    "FiEye","FiFileText","FiZap","FiGrid","FiCheckCircle","FiArrowRight","FiTrendingDown"];
  for (const n of names) { NAV[n] = await icon(n, NAVY); AMB[n] = await icon(n, AMBER); }

  const p = new pptxgen();
  p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
  p.layout = "W";
  const W = 13.333, H = 7.5;

  // helpers
  const bg = (s) => s.background = { color: NAVY };
  function panel(s, x, y, w, h, fill = PANEL, line = BORDER) {
    s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.09,
      fill: { color: fill }, line: { color: line, width: 1 } });
  }
  function iconCircle(s, x, y, d, img, ring = AMBER) {
    s.addShape(p.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: ring } });
    const pad = d * 0.24;
    s.addImage({ data: img, x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
  }
  function seg(s, x1, y1, x2, y2, color, w) {
    const dx = x2 - x1, dy = y2 - y1;
    s.addShape(p.ShapeType.line, { x: Math.min(x1, x2), y: Math.min(y1, y2),
      w: Math.abs(dx) || 0.001, h: Math.abs(dy) || 0.001, flipV: dx * dy < 0,
      line: { color, width: w } });
  }
  function dot(s, cx, cy, r, fill, line) {
    s.addShape(p.ShapeType.ellipse, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r,
      fill: { color: fill }, line: line ? { color: line, width: 1.5 } : { type: "none" } });
  }
  function ring(s, x, y, d, val, valColor) {
    s.addChart(p.ChartType.doughnut,
      [{ name: "r", labels: ["v", "rest"], values: [val, 100 - val] }],
      { x, y, w: d, h: d, holeSize: 74, showLegend: false, showTitle: false,
        showValue: false, chartColors: [valColor, BORDER], dataBorder: { pt: 0, color: NAVY } });
  }

  // ============================================================ SLIDE 1 - TITLE
  let s = p.addSlide(); bg(s);
  s.addText("LIVE CORPORATE RISK ENGINE", { x: 0.9, y: 1.15, w: 8, h: 0.4,
    fontFace: F, fontSize: 13, bold: true, color: AMBER, charSpacing: 4 });
  s.addText([
    { text: "Foresight ", options: { color: WHITE } },
    { text: "AI", options: { color: AMBER } },
  ], { x: 0.85, y: 1.55, w: 8.4, h: 1.4, fontFace: F, fontSize: 66, bold: true });
  s.addText("Point it at any listed Indian company. Get one explainable 0-100 score\nthat fuses the Altman Z-Score with the market signals others read separately:\ncredit ratings, leadership, news, hiring and employee sentiment.",
    { x: 0.9, y: 3.05, w: 7.9, h: 1.2, fontFace: F, fontSize: 18, color: DIM, lineSpacingMultiple: 1.12 });

  // link pill
  const heroLink = { url: "https://foresightai.streamlit.app", tooltip: "Open the live Foresight AI dashboard" };
  s.addShape(p.ShapeType.roundRect, { x: 0.9, y: 4.35, w: 4.55, h: 0.62, rectRadius: 0.31,
    fill: { color: PANEL2 }, line: { color: AMBER, width: 1.25 }, hyperlink: heroLink });
  s.addText([
    { text: "Live demo   ", options: { color: DIM, fontSize: 13 } },
    { text: "foresightai.streamlit.app", options: { color: AMBER, fontSize: 15, bold: true, underline: true } },
  ], { x: 0.9, y: 4.35, w: 4.55, h: 0.62, fontFace: F, align: "center", valign: "middle", hyperlink: heroLink });

  s.addText("Altman Z  +  credit ratings  +  leadership  +  news   .   NSE-listed   .   validated on 5 real collapses",
    { x: 0.9, y: 6.55, w: 9.5, h: 0.4, fontFace: F, fontSize: 12, color: DIM });

  // hero ring (right)
  const gx = 9.85, gy = 2.0, gd = 2.9;
  s.addShape(p.ShapeType.ellipse, { x: gx - 0.35, y: gy - 0.35, w: gd + 0.7, h: gd + 0.7,
    fill: { color: PANEL }, line: { color: BORDER, width: 1 } });
  ring(s, gx, gy, gd, 58, AMBER);
  s.addText("58", { x: gx, y: gy + 0.72, w: gd, h: 0.9, fontFace: F, fontSize: 52, bold: true,
    color: WHITE, align: "center" });
  s.addText("RISK / 100", { x: gx, y: gy + 1.62, w: gd, h: 0.35, fontFace: F, fontSize: 12,
    color: DIM, align: "center", charSpacing: 2 });
  s.addText("WATCH", { x: gx, y: gy - 0.02, w: gd, h: 0.3, fontFace: F, fontSize: 12, bold: true,
    color: AMBER, align: "center", charSpacing: 3 });
  s.addText("live, explainable, on demand", { x: gx - 0.35, y: gy + gd + 0.45, w: gd + 0.7, h: 0.35,
    fontFace: F, fontSize: 12.5, italic: true, color: DIM, align: "center" });

  // ============================================================ SLIDE 2 - WHY
  s = p.addSlide(); bg(s);
  s.addText([
    { text: "The warning signs are ", options: { color: WHITE } },
    { text: "public", options: { color: AMBER } },
    { text: ". Nobody reads them in time.", options: { color: WHITE } },
  ], { x: 0.85, y: 0.6, w: 11.7, h: 0.8, fontFace: F, fontSize: 33, bold: true });
  s.addText("Distress builds for years in the filings and the headlines - but the read is slow, manual and scattered.",
    { x: 0.87, y: 1.42, w: 11, h: 0.4, fontFace: F, fontSize: 15.5, color: DIM });

  const pains = [
    ["FiClock", "Backward-looking", "Financial statements lag reality by quarters. By the time the ratios turn, the market already knows."],
    ["FiSearch", "Manual, one by one", "Analysts read filings company by company. Coverage is thin and the slowest names get watched last."],
    ["FiRss", "Signals sit apart", "News moves before the accounts do - but the numbers and the narrative live in different places."],
  ];
  let cy = 2.15;
  pains.forEach(([ic, hd, tx]) => {
    panel(s, 0.85, cy, 6.55, 1.28);
    iconCircle(s, 1.12, cy + 0.32, 0.64, NAV[ic]);
    s.addText(hd, { x: 2.0, y: cy + 0.18, w: 5.2, h: 0.4, fontFace: F, fontSize: 17, bold: true, color: WHITE });
    s.addText(tx, { x: 2.0, y: cy + 0.55, w: 5.25, h: 0.66, fontFace: F, fontSize: 12.5, color: DIM, lineSpacingMultiple: 1.05 });
    cy += 1.48;
  });

  // RCom foreshadow callout (right)
  const rx = 7.75, rw = 4.75;
  panel(s, rx, 2.15, rw, 4.28, PANEL2);
  s.addText("THE FORESIGHT FIX", { x: rx + 0.35, y: 2.42, w: rw - 0.7, h: 0.35, fontFace: F,
    fontSize: 11.5, bold: true, color: AMBER, charSpacing: 3 });
  s.addText("Everything public, in one score", { x: rx + 0.35, y: 2.78, w: rw - 0.7, h: 0.4, fontFace: F,
    fontSize: 19, bold: true, color: WHITE });
  s.addText("6", { x: rx + 0.3, y: 3.32, w: 1.4, h: 1.1, fontFace: F, fontSize: 74, bold: true, color: AMBER, align: "left" });
  s.addText([
    { text: "signals", options: { bold: true, color: WHITE, fontSize: 20 } },
    { text: "\nfused into one\n0-100 read", options: { color: DIM, fontSize: 14 } },
  ], { x: rx + 1.7, y: 3.5, w: rw - 1.95, h: 1.0, fontFace: F, lineSpacingMultiple: 1.02 });
  s.addText("Financials  .  credit ratings  .  leadership  .  news  .  hiring  .  employee sentiment",
    { x: rx + 0.35, y: 4.95, w: rw - 0.7, h: 0.7, fontFace: F, fontSize: 12.5, color: WHITE, lineSpacingMultiple: 1.1 });
  s.addText("Read together and explained - not scattered across ten browser tabs.",
    { x: rx + 0.35, y: 5.95, w: rw - 0.7, h: 0.4, fontFace: F, fontSize: 11.5, italic: true, color: DIM });

  // ============================================================ SLIDE 3 - HOW (flow)
  s = p.addSlide(); bg(s);
  s.addText([
    { text: "One live pipeline, ", options: { color: WHITE } },
    { text: "fully explainable", options: { color: AMBER } },
  ], { x: 0.85, y: 0.55, w: 11.7, h: 0.7, fontFace: F, fontSize: 33, bold: true });
  s.addText("Nothing hard-coded. Pick a name and every step runs on the spot.",
    { x: 0.87, y: 1.34, w: 11, h: 0.4, fontFace: F, fontSize: 15.5, color: DIM });

  const nodes = [
    ["FiSearch", "Pick a company", "Any NSE-listed name, chosen on demand."],
    ["FiDownloadCloud", "Fetch live", "Financials from Screener + latest news from Google News."],
    ["FiActivity", "Score the signals", "Altman Z financials + a five-signal market pulse."],
    ["FiGitMerge", "Fuse + explain", "One 0-100 risk read, every point traced to a real ratio, rating or headline."],
  ];
  const nx0 = 0.65, nw = 2.55, gap = 0.72, step = nw + gap, ny = 2.15, nh = 2.55;
  nodes.forEach(([ic, hd, tx], i) => {
    const x = nx0 + i * step;
    panel(s, x, ny, nw, nh, i === 3 ? PANEL2 : PANEL, i === 3 ? AMBER : BORDER);
    iconCircle(s, x + nw / 2 - 0.42, ny + 0.32, 0.84, NAV[ic]);
    s.addText(`0${i + 1}`, { x: x + 0.18, y: ny + 0.16, w: 0.8, h: 0.35, fontFace: F, fontSize: 13, bold: true, color: AMBER });
    s.addText(hd, { x: x + 0.15, y: ny + 1.32, w: nw - 0.3, h: 0.4, fontFace: F, fontSize: 16.5, bold: true, color: WHITE, align: "center" });
    s.addText(tx, { x: x + 0.22, y: ny + 1.74, w: nw - 0.44, h: 0.7, fontFace: F, fontSize: 12, color: DIM, align: "center", lineSpacingMultiple: 1.03 });
    if (i < 3) s.addImage({ data: AMB.FiArrowRight, x: x + nw + 0.14, y: ny + nh / 2 - 0.22, w: 0.44, h: 0.44 });
  });

  // two-legs detail band
  const ly = 5.15;
  panel(s, 0.65, ly, W - 1.3, 1.55, PANEL);
  s.addText("INSIDE THE SCORE", { x: 0.95, y: ly + 0.22, w: 4, h: 0.3, fontFace: F, fontSize: 11, bold: true, color: AMBER, charSpacing: 3 });
  // leg 1
  iconCircle(s, 0.95, ly + 0.62, 0.6, NAV.FiFileText);
  s.addText([
    { text: "Financial leg", options: { bold: true, color: WHITE, fontSize: 14.5 } },
    { text: "   Original 1968 Altman Z - working capital, retained earnings, EBIT, market equity, sales. A trusted distress formula, computed live and decomposed term by term.", options: { color: DIM, fontSize: 12.5 } },
  ], { x: 1.7, y: ly + 0.6, w: 5.0, h: 0.85, fontFace: F, valign: "top", lineSpacingMultiple: 1.02 });
  // divider
  seg(s, 7.05, ly + 0.5, 7.05, ly + 1.25, BORDER, 1);
  // leg 2
  iconCircle(s, 7.35, ly + 0.62, 0.6, NAV.FiRss);
  s.addText([
    { text: "Market leg", options: { bold: true, color: WHITE, fontSize: 14.5 } },
    { text: "   Five signals that move before the accounts do: credit rating (0.30), leadership changes (0.25), news sentiment (0.25), hiring (0.10), employee confidence (0.10). A rating cut to 'D' or an auditor exit floors the score - facts, not moods.", options: { color: DIM, fontSize: 12 } },
  ], { x: 8.1, y: ly + 0.58, w: 4.55, h: 0.95, fontFace: F, valign: "top", lineSpacingMultiple: 1.0 });

  // ============================================================ SLIDE 4 - TRACK RECORD
  s = p.addSlide(); bg(s);
  s.addText([
    { text: "One ", options: { color: WHITE } },
    { text: "comprehensive", options: { color: AMBER } },
    { text: " score, on real history", options: { color: WHITE } },
  ], { x: 0.85, y: 0.55, w: 11.7, h: 0.7, fontFace: F, fontSize: 33, bold: true });
  s.addText("Altman reads the annual filings. ForesightAI reads the same financials plus the credit ratings, leadership, news and the market.",
    { x: 0.87, y: 1.34, w: 11.6, h: 0.4, fontFace: F, fontSize: 15.5, color: DIM });

  // left column - comprehensiveness
  s.addText("6 inputs", { x: 0.85, y: 2.05, w: 4.4, h: 0.75, fontFace: F, fontSize: 40, bold: true, color: AMBER });
  s.addText("financials, credit ratings, leadership, news, hiring and employee sentiment - fused into one 0-100 score.",
    { x: 0.87, y: 2.82, w: 4.35, h: 0.9, fontFace: F, fontSize: 13, color: WHITE, lineSpacingMultiple: 1.1 });
  s.addText("Altman is the financial leg. ForesightAI adds the market signals, so the score reflects the full public picture - and moves the day any of them do.",
    { x: 0.87, y: 3.72, w: 4.35, h: 1.15, fontFace: F, fontSize: 12, color: DIM, lineSpacingMultiple: 1.08 });
  panel(s, 0.85, 4.98, 4.35, 1.7, PANEL2);
  s.addText("FIVE CASES, FIVE SECTORS", { x: 1.08, y: 5.1, w: 3.9, h: 0.28, fontFace: F, fontSize: 10.5, bold: true, color: AMBER, charSpacing: 2 });
  s.addText([
    { text: "Unitech ", options: { bold: true, color: WHITE } }, { text: "(real estate), ", options: { color: DIM } },
    { text: "Future Retail ", options: { bold: true, color: WHITE } }, { text: "(retail), ", options: { color: DIM } },
    { text: "Reliance Communications ", options: { bold: true, color: WHITE } }, { text: "(telecom), ", options: { color: DIM } },
    { text: "Jaiprakash Associates ", options: { bold: true, color: WHITE } }, { text: "(infrastructure), ", options: { color: DIM } },
    { text: "Suzlon ", options: { bold: true, color: WHITE } }, { text: "(renewables - a recovery). Real financials, real events.", options: { color: DIM } },
  ], { x: 1.08, y: 5.38, w: 3.9, h: 1.24, fontFace: F, fontSize: 10.5, valign: "top", lineSpacingMultiple: 1.03 });

  // chart - stacked: Altman (financial) + added signal risk = the comprehensive score
  const PL = 5.85, PT = 2.4, PW = 6.85, PH = 3.95;
  const BLUE = "5B9BD5";
  const yrs4 = ["FY15", "FY16", "FY17", "FY18", "FY19", "FY20", "FY21", "FY22", "FY23"];
  const altman = [13, 20, 21, 29, 34, 49, 63, 69, 85];
  const added = [15, 28, 37, 45, 43, 34, 23, 19, 5];
  s.addText("Example - Unitech (real estate): Altman read Safe for years", { x: PL, y: 1.92, w: PW, h: 0.25, fontFace: F, fontSize: 11, color: DIM });
  s.addChart(p.ChartType.bar, [
    { name: "Altman (financial-only)", labels: yrs4, values: altman },
    { name: "Added risk: market signals", labels: yrs4, values: added },
  ], {
    x: PL, y: PT, w: PW, h: PH,
    barDir: "col", barGrouping: "stacked", barGapWidthPct: 45,
    chartColors: [BLUE, AMBER],
    showLegend: true, legendPos: "t", legendColor: WHITE, legendFontSize: 11,
    showValue: false, showTitle: false,
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25,
    catAxisLabelColor: DIM, catAxisLabelFontSize: 10, catAxisLineShow: false,
    valAxisLabelColor: DIM, valAxisLabelFontSize: 9, valAxisLineShow: false,
    valGridLine: { color: BORDER, size: 0.5 }, catGridLine: { style: "none" },
  });
  s.addText("The amber block is the risk Altman alone misses - only the comprehensive score shows it.",
    { x: PL, y: PT + PH + 0.06, w: PW, h: 0.3, fontFace: F, fontSize: 10.5, italic: true, color: DIM });

  // ============================================================ SLIDE 5 - KEY FINDINGS (measured)
  s = p.addSlide(); bg(s);
  s.addText([
    { text: "What the numbers ", options: { color: WHITE } },
    { text: "actually say", options: { color: AMBER } },
  ], { x: 0.85, y: 0.55, w: 11.7, h: 0.7, fontFace: F, fontSize: 33, bold: true });
  s.addText("Four findings, each reproducible from the notebooks - a measurement, not a story.",
    { x: 0.87, y: 1.34, w: 11.6, h: 0.4, fontFace: F, fontSize: 15.5, color: DIM });

  const finds = [
    ["FiActivity", "8 / 9", "caught at zero false alarms",
     "Across 12 healthy blue-chips and 9 firms that failed, a threshold that flags NO healthy name still catches 8 of the 9 failures. Median distress probability 0.83 vs 0.13 (leave-one-out ROC-AUC 0.97)."],
    ["FiClock", "27 months", "median early warning",
     "The comprehensive score crosses into high risk a median 27 months before insolvency - 6 months earlier than Altman. Unitech: 26 months, where Altman never warned in time."],
    ["FiZap", "76 -> 78", "the hard-event floor binds",
     "SpiceJet's verified auditor exit floors the signal pulse - a fact that softer, tone-based signals cannot average away. It stays inert on the healthy names."],
    ["FiGrid", "50/50 = 60/40 = 70/30", "robust to its weights",
     "The worst-to-best risk ranking of the six tracked names is identical across all three blends - the signals drive the read, not the exact weighting."],
  ];
  const fx0 = 0.85, fw = 5.85, fstep = fw + 0.75, fy0 = 2.05, fyStep = 2.28, fhh = 2.05;
  finds.forEach(([ic, stat, hd, tx], i) => {
    const col = i % 2, row = (i / 2) | 0;
    const x = fx0 + col * fstep, y = fy0 + row * fyStep;
    panel(s, x, y, fw, fhh, (row === 0 && col === 0) ? PANEL2 : PANEL, BORDER);
    iconCircle(s, x + 0.28, y + 0.28, 0.62, NAV[ic]);
    s.addText(stat, { x: x + 1.05, y: y + 0.18, w: fw - 1.25, h: 0.62, fontFace: F,
      fontSize: stat.length > 11 ? 19 : 30, bold: true, color: AMBER, valign: "middle" });
    s.addText(hd, { x: x + 0.3, y: y + 0.9, w: fw - 0.6, h: 0.3, fontFace: F, fontSize: 13, bold: true, color: WHITE });
    s.addText(tx, { x: x + 0.3, y: y + 1.2, w: fw - 0.6, h: 0.78, fontFace: F, fontSize: 11.5, color: DIM, lineSpacingMultiple: 1.05 });
  });
  s.addText([
    { text: "Reproducible:  ", options: { color: AMBER, bold: true } },
    { text: "notebooks/04_findings.ipynb  -  runs on the base requirements, no heavy ML deps.", options: { color: DIM } },
  ], { x: 0.87, y: 6.65, w: 11.6, h: 0.35, fontFace: F, fontSize: 12 });

  // ============================================================ SLIDE 6 - VALUE + CTA
  s = p.addSlide(); bg(s);
  s.addText([
    { text: "A live risk analyst for ", options: { color: WHITE } },
    { text: "every name in your book", options: { color: AMBER } },
  ], { x: 0.85, y: 0.6, w: 11.7, h: 0.75, fontFace: F, fontSize: 33, bold: true });
  s.addText("Faster credit reviews   .   earlier watch-list triage   .   one defensible number, on demand.",
    { x: 0.87, y: 1.44, w: 11.5, h: 0.4, fontFace: F, fontSize: 15.5, color: DIM });

  const tiles = [
    ["FiGrid", "Comprehensive", "Six inputs - financials, ratings, leadership, news, hiring, sentiment - in one score."],
    ["FiZap", "Live & automated", "Any listed company, on demand. No manual pull, no waiting on filings."],
    ["FiEye", "Explainable", "Every point traces to a real ratio, rating or headline."],
    ["FiCheckCircle", "Proven on real cases", "Replayed across five real collapses, in five sectors."],
  ];
  const tx0 = 0.85, tw = 2.78, tgap = 0.24, tstep = tw + tgap, ty = 2.25, th = 2.55;
  tiles.forEach(([ic, hd, tx], i) => {
    const x = tx0 + i * tstep;
    panel(s, x, ty, tw, th);
    iconCircle(s, x + 0.28, ty + 0.3, 0.7, NAV[ic]);
    s.addText(hd, { x: x + 0.24, y: ty + 1.12, w: tw - 0.44, h: 0.4, fontFace: F, fontSize: 15.5, bold: true, color: WHITE });
    s.addText(tx, { x: x + 0.24, y: ty + 1.5, w: tw - 0.44, h: 0.9, fontFace: F, fontSize: 12.5, color: DIM, lineSpacingMultiple: 1.06 });
  });

  // CTA row - a clickable live link to the deployed dashboard (works in slideshow)
  const APP_URL = "https://foresightai.streamlit.app";
  const appLink = { url: APP_URL, tooltip: "Open the live Foresight AI dashboard" };
  s.addShape(p.ShapeType.roundRect, { x: 0.85, y: 5.5, w: 6.2, h: 1.0, rectRadius: 0.5,
    fill: { color: AMBER }, hyperlink: appLink });
  s.addText([
    { text: "Open the live dashboard   ", options: { color: NAVY, fontSize: 17, bold: true } },
    { text: "foresightai.streamlit.app", options: { color: NAVY, fontSize: 19, bold: true, underline: true } },
  ], { x: 0.85, y: 5.5, w: 6.2, h: 1.0, fontFace: F, align: "center", valign: "middle", hyperlink: appLink });
  s.addText([
    { text: "Source & notebooks:  ", options: { color: DIM } },
    { text: "github.com/hiralsarkar/ForesightAI", options: { color: WHITE, bold: true, underline: true,
        hyperlink: { url: "https://github.com/hiralsarkar/ForesightAI", tooltip: "View the code on GitHub" } } },
  ], { x: 0.9, y: 6.62, w: 8, h: 0.35, fontFace: F, fontSize: 13 });

  // hero combined panel (right)
  const hx = 7.55, hw = 4.95;
  panel(s, hx, 5.5, hw, 1.0, PANEL2);
  s.addText("COMBINED READ", { x: hx + 0.3, y: 5.62, w: 3, h: 0.3, fontFace: F, fontSize: 10.5, bold: true, color: AMBER, charSpacing: 2 });
  // two mini bars
  const barL = hx + 0.3, barW = 2.6;
  s.addText("Financial", { x: barL, y: 5.92, w: 1.1, h: 0.25, fontFace: F, fontSize: 10.5, color: DIM });
  s.addShape(p.ShapeType.roundRect, { x: barL + 1.15, y: 5.95, w: barW, h: 0.16, rectRadius: 0.08, fill: { color: BORDER }, line: { type: "none" } });
  s.addShape(p.ShapeType.roundRect, { x: barL + 1.15, y: 5.95, w: barW * 0.62, h: 0.16, rectRadius: 0.08, fill: { color: AMBER }, line: { type: "none" } });
  s.addText("Market", { x: barL, y: 6.2, w: 1.1, h: 0.25, fontFace: F, fontSize: 10.5, color: DIM });
  s.addShape(p.ShapeType.roundRect, { x: barL + 1.15, y: 6.23, w: barW, h: 0.16, rectRadius: 0.08, fill: { color: BORDER }, line: { type: "none" } });
  s.addShape(p.ShapeType.roundRect, { x: barL + 1.15, y: 6.23, w: barW * 0.5, h: 0.16, rectRadius: 0.08, fill: { color: ORANGE }, line: { type: "none" } });
  // fused ring
  ring(s, hx + hw - 0.95, 5.6, 0.8, 58, AMBER);
  s.addText("58", { x: hx + hw - 0.95, y: 5.82, w: 0.8, h: 0.4, fontFace: F, fontSize: 18, bold: true, color: WHITE, align: "center" });

  await p.writeFile({ fileName: "ForesightAI.pptx" });
  console.log("written ForesightAI.pptx");
})();
