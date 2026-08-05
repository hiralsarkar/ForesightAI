// Foresight AI - 5 slide business-context deck. Dark navy / amber, matches the live dashboard.
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
  s.addText("Point it at any listed Indian company. Get a live, explainable\nrisk read in seconds - the Altman Z-Score fused with market signals:\ncredit ratings, leadership changes and live news.",
    { x: 0.9, y: 3.05, w: 7.7, h: 1.2, fontFace: F, fontSize: 18, color: DIM, lineSpacingMultiple: 1.12 });

  // link pill
  s.addShape(p.ShapeType.roundRect, { x: 0.9, y: 4.35, w: 4.55, h: 0.62, rectRadius: 0.31,
    fill: { color: PANEL2 }, line: { color: AMBER, width: 1.25 } });
  s.addText([
    { text: "Live demo   ", options: { color: DIM, fontSize: 13 } },
    { text: "foresightai.streamlit.app", options: { color: AMBER, fontSize: 15, bold: true } },
  ], { x: 0.9, y: 4.35, w: 4.55, h: 0.62, fontFace: F, align: "center", valign: "middle" });

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
  s.addText("SEEN IN HINDSIGHT", { x: rx + 0.35, y: 2.42, w: rw - 0.7, h: 0.35, fontFace: F,
    fontSize: 11.5, bold: true, color: AMBER, charSpacing: 3 });
  s.addText("Reliance Communications", { x: rx + 0.35, y: 2.78, w: rw - 0.7, h: 0.4, fontFace: F,
    fontSize: 19, bold: true, color: WHITE });
  s.addText("3", { x: rx + 0.3, y: 3.35, w: 1.6, h: 1.1, fontFace: F, fontSize: 74, bold: true, color: AMBER, align: "left" });
  s.addText([
    { text: "years", options: { bold: true, color: WHITE, fontSize: 20 } },
    { text: "\nof warning before\ninsolvency", options: { color: DIM, fontSize: 14 } },
  ], { x: rx + 1.85, y: 3.5, w: rw - 2.1, h: 1.0, fontFace: F, lineSpacingMultiple: 1.02 });
  // down sparkline
  const sp = [[rx + 0.4, 4.95], [rx + 1.25, 5.15], [rx + 2.1, 5.35], [rx + 2.95, 5.7], [rx + 3.9, 5.8]];
  for (let i = 0; i < sp.length - 1; i++) seg(s, sp[i][0], sp[i][1], sp[i + 1][0], sp[i + 1][1], BAD, 2.5);
  sp.forEach((pt, i) => dot(s, pt[0], pt[1], 0.055, i === 1 ? AMBER : BAD));
  s.addText("Altman Z entered distress in FY2016. Filed for insolvency in 2019.",
    { x: rx + 0.35, y: 6.02, w: rw - 0.7, h: 0.35, fontFace: F, fontSize: 11.5, italic: true, color: DIM });

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

  // ============================================================ SLIDE 4 - HINDSIGHT chart
  s = p.addSlide(); bg(s);
  s.addText([
    { text: "Would it have warned you? ", options: { color: WHITE } },
    { text: "Years early.", options: { color: AMBER } },
  ], { x: 0.85, y: 0.55, w: 11.7, h: 0.7, fontFace: F, fontSize: 33, bold: true });
  s.addText("Five real collapses, replayed - the Altman Z merged with the signals that led the accounts.",
    { x: 0.87, y: 1.34, w: 11.6, h: 0.4, fontFace: F, fontSize: 15.5, color: DIM });

  // left text
  s.addText("3 years", { x: 0.85, y: 2.05, w: 4.4, h: 0.8, fontFace: F, fontSize: 44, bold: true, color: AMBER });
  s.addText("of early warning on Reliance Communications.", { x: 0.87, y: 2.85, w: 4.3, h: 0.55, fontFace: F, fontSize: 15.5, bold: true, color: WHITE });
  s.addText("Altman Z slid into distress by FY2016 - and the rating cut, the Ericsson insolvency plea and the news cascade all pointed the same way, three years before the 2019 filing.",
    { x: 0.87, y: 3.4, w: 4.35, h: 1.35, fontFace: F, fontSize: 12.5, color: DIM, lineSpacingMultiple: 1.08 });
  panel(s, 0.85, 4.95, 4.35, 1.72, PANEL2);
  s.addText("FIVE CASES, FIVE DYNAMICS", { x: 1.08, y: 5.08, w: 3.9, h: 0.28, fontFace: F, fontSize: 10.5, bold: true, color: AMBER, charSpacing: 2 });
  s.addText([
    { text: "RCom & Suzlon ", options: { bold: true, color: WHITE } },
    { text: "- Altman leads (Suzlon even recovers).  ", options: { color: DIM } },
    { text: "DHFL & IL&FS ", options: { bold: true, color: WHITE } },
    { text: "- NBFCs where Altman is blind; IL&FS was rated AAA and only the signals led.  ", options: { color: DIM } },
    { text: "Manpasand ", options: { bold: true, color: WHITE } },
    { text: "- an auditor exit, 12 months early.", options: { color: DIM } },
  ], { x: 1.08, y: 5.36, w: 3.9, h: 1.24, fontFace: F, fontSize: 10.5, valign: "top", lineSpacingMultiple: 1.03 });

  // chart geometry
  const PL = 5.85, PT = 2.15, PW = 6.85, PH = 4.4, PB = PT + PH, PR = PL + PW;
  const vTop = 3.0, vBot = -3.6, vr = vTop - vBot;
  const yv = (v) => PT + (vTop - v) / vr * PH;
  const years = [2015, 2016, 2017, 2018, 2019];
  const xv = (i) => PL + (i / (years.length - 1)) * PW;
  const rcom = [1.98, 0.53, -0.37, -2.81, -3.11];
  // zone bands
  s.addShape(p.ShapeType.rect, { x: PL, y: PT, w: PW, h: yv(2.6) - PT, fill: { color: GOOD, transparency: 88 }, line: { type: "none" } });
  s.addShape(p.ShapeType.rect, { x: PL, y: yv(2.6), w: PW, h: yv(1.1) - yv(2.6), fill: { color: DIM, transparency: 90 }, line: { type: "none" } });
  s.addShape(p.ShapeType.rect, { x: PL, y: yv(1.1), w: PW, h: PB - yv(1.1), fill: { color: BAD, transparency: 84 }, line: { type: "none" } });
  // threshold lines + labels
  seg(s, PL, yv(2.6), PR, yv(2.6), GOOD, 1);
  seg(s, PL, yv(1.1), PR, yv(1.1), BAD, 1);
  s.addText("SAFE  Z'' > 2.6", { x: PL + 0.12, y: PT + 0.02, w: 2, h: 0.25, fontFace: F, fontSize: 10, bold: true, color: GOOD, align: "left" });
  s.addText("DISTRESS  Z'' < 1.1", { x: PL + 0.12, y: yv(1.1) + 0.06, w: 2, h: 0.25, fontFace: F, fontSize: 10, bold: true, color: BAD, align: "left" });
  // axes baseline
  seg(s, PL, PB, PR, PB, BORDER, 1);
  // event marker (2019 insolvency)
  const exx = xv(4);
  for (let yy = PT; yy < PB; yy += 0.28) seg(s, exx, yy, exx, Math.min(yy + 0.14, PB), AMBER, 1.5);
  s.addText("INSOLVENCY 2019", { x: exx - 1.9, y: PT - 0.05, w: 1.85, h: 0.25, fontFace: F, fontSize: 10, bold: true, color: AMBER, align: "right" });
  // rcom line
  for (let i = 0; i < rcom.length - 1; i++) seg(s, xv(i), yv(rcom[i]), xv(i + 1), yv(rcom[i + 1]), AMBER, 3);
  rcom.forEach((v, i) => {
    const hi = i === 1;
    dot(s, xv(i), yv(v), hi ? 0.11 : 0.075, hi ? WHITE : AMBER, hi ? AMBER : NAVY);
  });
  // year labels
  years.forEach((yr, i) => s.addText("FY" + String(yr).slice(2), { x: xv(i) - 0.4, y: PB + 0.06, w: 0.8, h: 0.28, fontFace: F, fontSize: 11, color: DIM, align: "center" }));
  // callout at 2016
  s.addText("distress flagged FY16", { x: xv(1) - 0.35, y: yv(0.53) + 0.18, w: 2.2, h: 0.3, fontFace: F, fontSize: 10.5, bold: true, color: WHITE });

  // signal-cascade markers on the timeline + legend (the merged signals, not just Altman)
  const BLUE = "3B82F6", PURPLE = "A855F7";
  const evs = [[2, ORANGE, "rating cut to default"], [3, BLUE, "Ericsson insolvency plea"]];
  evs.forEach(([i, col]) => dot(s, xv(i), yv(rcom[i]), 0.075, col, WHITE));
  const lgx = PR - 2.05, lgy = PT + 0.12;
  s.addText("SIGNALS MERGED", { x: lgx, y: lgy - 0.02, w: 2.0, h: 0.24, fontFace: F, fontSize: 9, bold: true, color: DIM, charSpacing: 1.5 });
  [["Credit rating", ORANGE], ["Leadership", BLUE], ["News", DIM], ["Auditor", PURPLE]].forEach(([lab, col], k) => {
    const yy = lgy + 0.28 + k * 0.26;
    dot(s, lgx + 0.08, yy + 0.08, 0.055, col);
    s.addText(lab, { x: lgx + 0.24, y: yy - 0.06, w: 1.8, h: 0.26, fontFace: F, fontSize: 10, color: WHITE });
  });

  // ============================================================ SLIDE 5 - VALUE + CTA
  s = p.addSlide(); bg(s);
  s.addText([
    { text: "A live risk analyst for ", options: { color: WHITE } },
    { text: "every name in your book", options: { color: AMBER } },
  ], { x: 0.85, y: 0.6, w: 11.7, h: 0.75, fontFace: F, fontSize: 33, bold: true });
  s.addText("Faster credit reviews   .   earlier watch-list triage   .   one defensible number, on demand.",
    { x: 0.87, y: 1.44, w: 11.5, h: 0.4, fontFace: F, fontSize: 15.5, color: DIM });

  const tiles = [
    ["FiZap", "Live & automated", "Any company, on demand. No manual pull, no waiting on filings."],
    ["FiGrid", "Comprehensive", "Altman Z fused with credit ratings, leadership, news and hiring."],
    ["FiEye", "Explainable", "Every point traces to a real ratio or a real headline."],
    ["FiCheckCircle", "Validated on history", "Flags real bankruptcies years before the event."],
  ];
  const tx0 = 0.85, tw = 2.78, tgap = 0.24, tstep = tw + tgap, ty = 2.25, th = 2.55;
  tiles.forEach(([ic, hd, tx], i) => {
    const x = tx0 + i * tstep;
    panel(s, x, ty, tw, th);
    iconCircle(s, x + 0.28, ty + 0.3, 0.7, NAV[ic]);
    s.addText(hd, { x: x + 0.24, y: ty + 1.12, w: tw - 0.44, h: 0.4, fontFace: F, fontSize: 15.5, bold: true, color: WHITE });
    s.addText(tx, { x: x + 0.24, y: ty + 1.5, w: tw - 0.44, h: 0.9, fontFace: F, fontSize: 12.5, color: DIM, lineSpacingMultiple: 1.06 });
  });

  // CTA row
  s.addShape(p.ShapeType.roundRect, { x: 0.85, y: 5.5, w: 6.2, h: 1.0, rectRadius: 0.5,
    fill: { color: AMBER } });
  s.addText([
    { text: "Try it live   ", options: { color: NAVY, fontSize: 18, bold: true } },
    { text: "foresightai.streamlit.app", options: { color: NAVY, fontSize: 20, bold: true } },
  ], { x: 0.85, y: 5.5, w: 6.2, h: 1.0, fontFace: F, align: "center", valign: "middle" });
  s.addText([
    { text: "Source & notebooks:  ", options: { color: DIM } },
    { text: "github.com/hiralsarkar/ForesightAI", options: { color: WHITE, bold: true } },
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
