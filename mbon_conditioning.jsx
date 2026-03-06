import { useState, useEffect, useRef } from "react";

const NUM_KC = 40;
const KC_ACTIVE = 4;
const NUM_MBON_APPROACH = 5;
const NUM_MBON_AVOID = 5;

function seededRand(seed) {
  let s = seed;
  return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
}

const rngA = seededRand(777);
const KC_ACTIVE_ODOR_A = [];
const usedA = new Set();
while (KC_ACTIVE_ODOR_A.length < KC_ACTIVE) {
  const idx = Math.floor(rngA() * NUM_KC);
  if (!usedA.has(idx)) { usedA.add(idx); KC_ACTIVE_ODOR_A.push(idx); }
}
const rngB = seededRand(1234);
const KC_ACTIVE_ODOR_B = [];
const usedB = new Set();
while (KC_ACTIVE_ODOR_B.length < KC_ACTIVE) {
  const idx = Math.floor(rngB() * NUM_KC);
  if (!usedB.has(idx) && !usedA.has(idx)) { usedB.add(idx); KC_ACTIVE_ODOR_B.push(idx); }
  if (usedB.size + usedA.size >= NUM_KC) {
    const idx2 = Math.floor(rngB() * NUM_KC);
    if (!usedB.has(idx2)) { usedB.add(idx2); KC_ACTIVE_ODOR_B.push(idx2); }
  }
}

const ODOR_A = { id: "A", name: "Odor A (octanol)", color: "#f0c040", indices: KC_ACTIVE_ODOR_A };
const ODOR_B = { id: "B", name: "Odor B (MCH)", color: "#c084fc", indices: KC_ACTIVE_ODOR_B };

const kcToApproach = Array.from({ length: NUM_MBON_APPROACH }, (_, mi) => {
  const r = seededRand(mi * 31 + 100);
  return Array.from({ length: NUM_KC }, () => 0.3 + r() * 0.7);
});
const kcToAvoid = Array.from({ length: NUM_MBON_AVOID }, (_, mi) => {
  const r = seededRand(mi * 53 + 200);
  return Array.from({ length: NUM_KC }, () => 0.3 + r() * 0.7);
});

// Stages now include training phases
const STAGES = [
  { id: "naive", label: "Naïve", desc: "No conditioning. Both MBON populations respond equally to any odor.",
    wA: { ap: 1.0, av: 1.0 }, wB: { ap: 1.0, av: 1.0 } },
  { id: "aversive_train", label: "Aversive Training", desc: "Odor A + electric shock. PPL1 DANs activate and depress KC→Approach MBON synapses in real time.",
    wA: { ap: 1.0, av: 1.0 }, wB: { ap: 1.0, av: 1.0 }, training: "aversive" },
  { id: "post_aversive", label: "Post-Aversive", desc: "After aversive conditioning. Odor A's approach synapses are depressed. Odor B is unaffected.",
    wA: { ap: 0.15, av: 1.0 }, wB: { ap: 1.0, av: 1.0 } },
  { id: "appetitive_train", label: "Appetitive Training", desc: "Odor B + sugar reward. PAM DANs activate and depress KC→Avoidance MBON synapses in real time.",
    wA: { ap: 0.15, av: 1.0 }, wB: { ap: 1.0, av: 1.0 }, training: "appetitive" },
  { id: "post_appetitive", label: "Post-Appetitive", desc: "After both conditionings. Odor A → avoidance. Odor B → approach. Toggle to compare.",
    wA: { ap: 0.15, av: 1.0 }, wB: { ap: 1.0, av: 0.15 } },
];

// Auto-play sequence
const AUTO_SEQ = [
  { stage: 0, a: true,  b: false, dur: 3000, label: "Naïve — Odor A" },
  { stage: 0, a: false, b: true,  dur: 3000, label: "Naïve — Odor B" },
  { stage: 0, a: false, b: false, dur: 1000, label: "Naïve — baseline" },
  { stage: 1, a: true,  b: false, dur: 5500, label: "Aversive Training — Odor A + Shock (PPL1 active)" },
  { stage: 2, a: true,  b: false, dur: 3500, label: "Post-Aversive — Odor A (conditioned → avoid)" },
  { stage: 2, a: false, b: true,  dur: 3000, label: "Post-Aversive — Odor B (unaffected)" },
  { stage: 2, a: false, b: false, dur: 1000, label: "Transition" },
  { stage: 3, a: false, b: true,  dur: 5500, label: "Appetitive Training — Odor B + Sugar (PAM active)" },
  { stage: 4, a: false, b: true,  dur: 3500, label: "Post-Both — Odor B (conditioned → approach)" },
  { stage: 4, a: true,  b: false, dur: 3500, label: "Post-Both — Odor A (conditioned → avoid)" },
  { stage: 4, a: true,  b: true,  dur: 3500, label: "Post-Both — Both odors" },
];

export default function MBONConditioning() {
  const [stageIdx, setStageIdx] = useState(0);
  const [odorAOn, setOdorAOn] = useState(false);
  const [odorBOn, setOdorBOn] = useState(false);
  const [autoPlay, setAutoPlay] = useState(true);
  const [autoStep, setAutoStep] = useState(0);
  const [autoLabel, setAutoLabel] = useState(AUTO_SEQ[0].label);

  // Animated values
  const [kcGlowA, setKcGlowA] = useState(0);
  const [kcGlowB, setKcGlowB] = useState(0);
  const [approachFiring, setApproachFiring] = useState(Array(NUM_MBON_APPROACH).fill(0));
  const [avoidFiring, setAvoidFiring] = useState(Array(NUM_MBON_AVOID).fill(0));

  // DAN and training animation state
  const [danPPL1, setDanPPL1] = useState(0);
  const [danPAM, setDanPAM] = useState(0);
  const [animWeightAap, setAnimWeightAap] = useState(1.0);
  const [animWeightAav, setAnimWeightAav] = useState(1.0);
  const [animWeightBap, setAnimWeightBap] = useState(1.0);
  const [animWeightBav, setAnimWeightBav] = useState(1.0);

  const animRef = useRef(null);
  const curA = useRef(0);
  const curB = useRef(0);
  const autoTimerRef = useRef(null);
  const trainStartRef = useRef(null);

  const stage = STAGES[stageIdx];
  const isTraining = !!stage.training;

  // When stage changes (non-training), snap animated weights to stage weights
  useEffect(() => {
    if (!isTraining) {
      setAnimWeightAap(stage.wA.ap);
      setAnimWeightAav(stage.wA.av);
      setAnimWeightBap(stage.wB.ap);
      setAnimWeightBav(stage.wB.av);
      setDanPPL1(0);
      setDanPAM(0);
      trainStartRef.current = null;
    } else {
      // Training starts: set initial weights and record start time
      setAnimWeightAap(stage.wA.ap);
      setAnimWeightAav(stage.wA.av);
      setAnimWeightBap(stage.wB.ap);
      setAnimWeightBav(stage.wB.av);
      trainStartRef.current = performance.now();
    }
  }, [stageIdx]);

  // Current effective weights (animated during training, static otherwise)
  const wAap = animWeightAap;
  const wAav = animWeightAav;
  const wBap = animWeightBap;
  const wBav = animWeightBav;

  // Auto-play
  useEffect(() => {
    if (!autoPlay) {
      if (autoTimerRef.current) clearTimeout(autoTimerRef.current);
      return;
    }
    const applyStep = (stepIdx) => {
      const step = AUTO_SEQ[stepIdx];
      setStageIdx(step.stage);
      setOdorAOn(step.a);
      setOdorBOn(step.b);
      setAutoLabel(step.label);
      setAutoStep(stepIdx);
      autoTimerRef.current = setTimeout(() => {
        applyStep((stepIdx + 1) % AUTO_SEQ.length);
      }, step.dur);
    };
    applyStep(autoStep);
    return () => { if (autoTimerRef.current) clearTimeout(autoTimerRef.current); };
  }, [autoPlay]);

  const goManual = () => setAutoPlay(false);

  // Main animation loop
  useEffect(() => {
    let running = true;
    const animate = () => {
      if (!running) return;

      // Smooth odor glow
      const tA = odorAOn ? 1 : 0;
      const tB = odorBOn ? 1 : 0;
      curA.current += (tA - curA.current) * 0.08;
      curB.current += (tB - curB.current) * 0.08;
      if (Math.abs(curA.current - tA) < 0.005) curA.current = tA;
      if (Math.abs(curB.current - tB) < 0.005) curB.current = tB;
      const cA = curA.current;
      const cB = curB.current;
      setKcGlowA(cA);
      setKcGlowB(cB);

      // Training animation
      const now = performance.now();
      if (isTraining && trainStartRef.current) {
        const elapsed = (now - trainStartRef.current) / 1000; // seconds
        const trainDur = 4.5; // seconds of active training
        const rampUp = 0.5;
        const rampDown = 0.5;

        let danStrength = 0;
        if (elapsed < rampUp) {
          danStrength = elapsed / rampUp;
        } else if (elapsed < rampUp + trainDur) {
          danStrength = 1.0;
        } else if (elapsed < rampUp + trainDur + rampDown) {
          danStrength = 1.0 - (elapsed - rampUp - trainDur) / rampDown;
        }
        danStrength = Math.max(0, Math.min(1, danStrength));

        // Pulsing effect on DAN
        const pulse = danStrength * (0.7 + 0.3 * Math.sin(now * 0.008));

        // Weight depression progress
        const depressionProgress = Math.min(1, Math.max(0, (elapsed - rampUp) / trainDur));
        const depressedWeight = 1.0 - depressionProgress * 0.85;

        if (stage.training === "aversive") {
          setDanPPL1(pulse);
          setDanPAM(0);
          setAnimWeightAap(Math.max(0.15, depressedWeight));
        } else if (stage.training === "appetitive") {
          setDanPPL1(0);
          setDanPAM(pulse);
          setAnimWeightBav(Math.max(0.15, depressedWeight));
        }
      }

      // MBON firing with current animated weights
      const curWAap = isTraining ? animWeightAap : stage.wA.ap;
      const curWAav = isTraining ? animWeightAav : stage.wA.av;
      const curWBap = isTraining ? animWeightBap : stage.wB.ap;
      const curWBav = isTraining ? animWeightBav : stage.wB.av;

      const aFire = Array.from({ length: NUM_MBON_APPROACH }, (_, i) => {
        let sum = 0;
        if (cA > 0.01) sum += cA * (ODOR_A.indices.reduce((s, ki) => s + kcToApproach[i][ki], 0) / KC_ACTIVE) * curWAap;
        if (cB > 0.01) sum += cB * (ODOR_B.indices.reduce((s, ki) => s + kcToApproach[i][ki], 0) / KC_ACTIVE) * curWBap;
        return Math.min(sum, 1);
      });
      const vFire = Array.from({ length: NUM_MBON_AVOID }, (_, i) => {
        let sum = 0;
        if (cA > 0.01) sum += cA * (ODOR_A.indices.reduce((s, ki) => s + kcToAvoid[i][ki], 0) / KC_ACTIVE) * curWAav;
        if (cB > 0.01) sum += cB * (ODOR_B.indices.reduce((s, ki) => s + kcToAvoid[i][ki], 0) / KC_ACTIVE) * curWBav;
        return Math.min(sum, 1);
      });
      setApproachFiring(aFire);
      setAvoidFiring(vFire);

      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => { running = false; if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [stageIdx, odorAOn, odorBOn, isTraining]);

  const approachMean = approachFiring.reduce((a, b) => a + b, 0) / NUM_MBON_APPROACH;
  const avoidMean = avoidFiring.reduce((a, b) => a + b, 0) / NUM_MBON_AVOID;
  const netDrive = approachMean - avoidMean;

  const APPROACH_COL = "#22d68a";
  const AVOID_COL = "#f5425a";
  const DAN_PPL1_COL = "#ff8844";
  const DAN_PAM_COL = "#44aaff";
  const BG = "#06060e";
  const PANEL = "#0c0c18";
  const BORDER = "#1a1a2e";

  function hexToRgb(hex) {
    return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
  }
  function lerp(a, b, t) { return a + (b - a) * Math.max(0, Math.min(1, t)); }

  function cellGlow(cx, cy, r, activity, color, label, key) {
    const rgb = hexToRgb(color);
    const fR = Math.round(lerp(14, rgb[0], activity));
    const fG = Math.round(lerp(14, rgb[1], activity));
    const fB = Math.round(lerp(24, rgb[2], activity));
    return (
      <g key={key}>
        {activity > 0.1 && <circle cx={cx} cy={cy} r={r + 10} fill={color} opacity={activity * 0.12} />}
        <circle cx={cx} cy={cy} r={r}
          fill={`rgb(${fR},${fG},${fB})`}
          stroke={activity > 0.15 ? color : "#1e1e32"}
          strokeWidth={activity > 0.15 ? 2 : 0.8} />
        {label && <text x={cx} y={cy + 3.5} textAnchor="middle" fontSize={8}
          fill={activity > 0.3 ? "#fff" : "#3a3a55"} fontFamily="inherit">{label}</text>}
      </g>
    );
  }

  // DAN neuron with animated pulse rings
  function danNeuron(cx, cy, activity, color, label) {
    if (activity < 0.02) return null;
    const rgb = hexToRgb(color);
    return (
      <g>
        {/* Outer pulse ring */}
        <circle cx={cx} cy={cy} r={22} fill="none"
          stroke={color} strokeWidth={1.5} opacity={activity * 0.25}
          strokeDasharray="4 3" />
        {/* Middle glow */}
        <circle cx={cx} cy={cy} r={16} fill={color} opacity={activity * 0.15} />
        {/* Core */}
        <circle cx={cx} cy={cy} r={12}
          fill={`rgba(${rgb[0]},${rgb[1]},${rgb[2]},${activity * 0.7})`}
          stroke={color} strokeWidth={2 * activity} />
        <text x={cx} y={cy - 1} textAnchor="middle" fontSize={7}
          fill="#fff" fontFamily="inherit" opacity={activity}>DAN</text>
        <text x={cx} y={cy + 9} textAnchor="middle" fontSize={6}
          fill={color} fontFamily="inherit" opacity={activity * 0.8}>{label}</text>
      </g>
    );
  }

  // Modulation arrow from DAN to MBON group
  function danModArrow(fx, fy, tx, ty, activity, color) {
    if (activity < 0.02) return null;
    const mx = (fx + tx) / 2;
    const my = (fy + ty) / 2 - 15;
    return (
      <path d={`M${fx},${fy + 14} Q${mx},${my} ${tx},${ty}`}
        fill="none" stroke={color} strokeWidth={2.5 * activity}
        strokeDasharray="6 4" opacity={activity * 0.7} />
    );
  }

  const svgW = 700, svgH = 440;
  const kcY = 80;
  const mbonY = 280;
  const danY = 175;
  const approachX = 180;
  const avoidX = svgW - 180;

  const kcPositions = Array.from({ length: NUM_KC }, (_, i) => ({
    x: 100 + i * ((svgW - 200) / (NUM_KC - 1)),
    y: kcY + Math.sin((i / (NUM_KC - 1)) * Math.PI) * 15,
  }));
  const approachPositions = Array.from({ length: NUM_MBON_APPROACH }, (_, i) => ({
    x: approachX - 40 + (i % 3) * 40, y: mbonY + Math.floor(i / 3) * 44,
  }));
  const avoidPositions = Array.from({ length: NUM_MBON_AVOID }, (_, i) => ({
    x: avoidX - 40 + (i % 3) * 40, y: mbonY + Math.floor(i / 3) * 44,
  }));

  function synapseLines(odor, glowVal, mbonPos, weightVal, color, prefix) {
    if (glowVal < 0.02) return null;
    return odor.indices.map((ki) => {
      const kc = kcPositions[ki];
      return mbonPos.map((mp, mi) => {
        const op = glowVal * weightVal * 0.25;
        if (op < 0.03) return null;
        return (
          <line key={`${prefix}-${ki}-${mi}`}
            x1={kc.x} y1={kc.y + 12} x2={mp.x} y2={mp.y - 18}
            stroke={color} strokeWidth={weightVal * 2.5 + 0.3}
            opacity={op}
            strokeDasharray={weightVal < 0.4 ? "2 4" : "none"} />
        );
      });
    });
  }

  const btnBase = {
    borderRadius: 8, padding: "6px 16px", fontSize: 12,
    fontFamily: "inherit", cursor: "pointer", letterSpacing: "0.04em",
    fontWeight: 600, transition: "all 0.2s",
  };

  return (
    <div style={{
      background: BG, minHeight: "100vh",
      fontFamily: "'IBM Plex Mono', 'Fira Code', 'Courier New', monospace",
      color: "#c0c0d0", display: "flex", flexDirection: "column",
      alignItems: "center", padding: "20px 8px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet" />

      <h1 style={{
        fontSize: 20, fontWeight: 700, letterSpacing: "0.1em",
        color: "#e4e4f0", margin: 0, textTransform: "uppercase", textAlign: "center",
      }}>MBON Valence Coding & Conditioning</h1>
      <p style={{ fontSize: 10, color: "#555570", letterSpacing: "0.14em", margin: "4px 0 14px", textAlign: "center" }}>
        Dopaminergic modulation of KC→MBON synapses during learning
      </p>

      {/* Stage selector */}
      <div style={{ marginBottom: 8, textAlign: "center" }}>
        <div style={{ fontSize: 9, color: "#555570", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 6 }}>
          Conditioning Stage
        </div>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap", justifyContent: "center" }}>
          {STAGES.map((s, i) => (
            <button key={i} onClick={() => { goManual(); setStageIdx(i); }}
              style={{
                ...btnBase, fontSize: 9, padding: "4px 10px",
                background: stageIdx === i ? (s.training ? (s.training === "aversive" ? DAN_PPL1_COL + "18" : DAN_PAM_COL + "18") : "#ffffff0e") : "transparent",
                border: `1.5px solid ${stageIdx === i ? (s.training ? (s.training === "aversive" ? DAN_PPL1_COL : DAN_PAM_COL) : "#888") : "#252538"}`,
                color: stageIdx === i ? (s.training ? (s.training === "aversive" ? DAN_PPL1_COL : DAN_PAM_COL) : "#e0e0f0") : "#505068",
              }}>
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Odor toggles + auto */}
      <div style={{ display: "flex", gap: 10, marginBottom: 8, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
        <div style={{ fontSize: 9, color: "#555570", letterSpacing: "0.12em", textTransform: "uppercase" }}>Stimulus</div>
        <button onClick={() => { goManual(); setOdorAOn(!odorAOn); }}
          style={{ ...btnBase, background: odorAOn ? ODOR_A.color + "20" : "transparent",
            border: `2px solid ${odorAOn ? ODOR_A.color : "#2a2a3e"}`, color: odorAOn ? ODOR_A.color : "#505068" }}>
          {odorAOn ? "● " : "○ "}{ODOR_A.name}
        </button>
        <button onClick={() => { goManual(); setOdorBOn(!odorBOn); }}
          style={{ ...btnBase, background: odorBOn ? ODOR_B.color + "20" : "transparent",
            border: `2px solid ${odorBOn ? ODOR_B.color : "#2a2a3e"}`, color: odorBOn ? ODOR_B.color : "#505068" }}>
          {odorBOn ? "● " : "○ "}{ODOR_B.name}
        </button>
        <button onClick={() => { setAutoPlay(!autoPlay); if (!autoPlay) setAutoStep(0); }}
          style={{ ...btnBase, background: autoPlay ? "#ffffff10" : "transparent",
            border: `1.5px solid ${autoPlay ? "#666" : "#2a2a3e"}`, color: autoPlay ? "#bbb" : "#505068" }}>
          {autoPlay ? "⏸ Auto" : "▶ Auto"}
        </button>
      </div>

      {/* Auto label / description */}
      <div style={{
        background: PANEL, border: `1px solid ${isTraining ? (stage.training === "aversive" ? DAN_PPL1_COL + "44" : DAN_PAM_COL + "44") : BORDER}`,
        borderRadius: 8, padding: "6px 18px", marginBottom: 12, maxWidth: 680, textAlign: "center",
        transition: "border-color 0.5s",
      }}>
        {autoPlay && <div style={{ fontSize: 9, color: "#6868a0", marginBottom: 2, fontStyle: "italic" }}>▸ {autoLabel}</div>}
        <span style={{ fontSize: 10, color: "#8888a0", lineHeight: 1.5 }}>{stage.desc}</span>
      </div>

      {/* Main SVG */}
      <div style={{ background: PANEL, border: `1px solid ${BORDER}`, borderRadius: 14, padding: "14px 10px" }}>
        <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} style={{ display: "block", maxWidth: "90vw" }}>
          <defs>
            <marker id="netArrowG" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill={APPROACH_COL} opacity="0.8" />
            </marker>
            <marker id="netArrowR" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill={AVOID_COL} opacity="0.8" />
            </marker>
            {/* X marker for depressed synapses */}
            <marker id="xMark" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
              <line x1="1" y1="1" x2="7" y2="7" stroke="#ff4444" strokeWidth="1.5" />
              <line x1="7" y1="1" x2="1" y2="7" stroke="#ff4444" strokeWidth="1.5" />
            </marker>
          </defs>

          {/* KC label */}
          <text x={svgW / 2} y={kcY - 40} textAnchor="middle" fontSize={10}
            fill="#6a6a80" fontFamily="inherit" letterSpacing="0.12em">KENYON CELLS</text>
          {odorAOn && <text x={svgW / 2 - (odorBOn ? 80 : 0)} y={kcY - 26} textAnchor="middle" fontSize={9}
            fill={ODOR_A.color} fontFamily="inherit" opacity={0.85}>{ODOR_A.name}</text>}
          {odorBOn && <text x={svgW / 2 + (odorAOn ? 80 : 0)} y={kcY - 26} textAnchor="middle" fontSize={9}
            fill={ODOR_B.color} fontFamily="inherit" opacity={0.85}>{ODOR_B.name}</text>}

          {/* MBON labels */}
          <text x={approachX} y={mbonY - 32} textAnchor="middle" fontSize={10}
            fill={APPROACH_COL} fontFamily="inherit" letterSpacing="0.1em" opacity={0.7}>APPROACH MBONs</text>
          <text x={avoidX} y={mbonY - 32} textAnchor="middle" fontSize={10}
            fill={AVOID_COL} fontFamily="inherit" letterSpacing="0.1em" opacity={0.7}>AVOIDANCE MBONs</text>

          {/* Synapse lines with animated weights */}
          {synapseLines(ODOR_A, kcGlowA, approachPositions, wAap, APPROACH_COL, "apA")}
          {synapseLines(ODOR_A, kcGlowA, avoidPositions, wAav, AVOID_COL, "avA")}
          {synapseLines(ODOR_B, kcGlowB, approachPositions, wBap, APPROACH_COL, "apB")}
          {synapseLines(ODOR_B, kcGlowB, avoidPositions, wBav, AVOID_COL, "avB")}

          {/* DAN modulation arrows — visible during training AND as indicators after */}
          {danModArrow(svgW / 2 - 80, danY, approachX + 30, mbonY - 24, danPPL1, DAN_PPL1_COL)}
          {danModArrow(svgW / 2 + 80, danY, avoidX - 30, mbonY - 24, danPAM, DAN_PAM_COL)}

          {/* Static modification indicators for post-conditioning stages */}
          {!isTraining && (stage.wA.ap < 1 || stage.wB.ap < 1) && (
            <g opacity={0.45}>
              <text x={svgW / 2 - 80} y={danY + 4} textAnchor="middle" fontSize={7}
                fill={DAN_PPL1_COL} fontFamily="inherit">PPL1 modified</text>
              <line x1={svgW / 2 - 80} y1={danY + 8} x2={approachX + 20} y2={mbonY - 24}
                stroke={DAN_PPL1_COL} strokeWidth={1} strokeDasharray="3 5" opacity={0.35} />
            </g>
          )}
          {!isTraining && (stage.wA.av < 1 || stage.wB.av < 1) && (
            <g opacity={0.45}>
              <text x={svgW / 2 + 80} y={danY + 4} textAnchor="middle" fontSize={7}
                fill={DAN_PAM_COL} fontFamily="inherit">PAM modified</text>
              <line x1={svgW / 2 + 80} y1={danY + 8} x2={avoidX - 20} y2={mbonY - 24}
                stroke={DAN_PAM_COL} strokeWidth={1} strokeDasharray="3 5" opacity={0.35} />
            </g>
          )}

          {/* DAN neurons — animated */}
          {danNeuron(svgW / 2 - 80, danY, danPPL1, DAN_PPL1_COL, "PPL1")}
          {danNeuron(svgW / 2 + 80, danY, danPAM, DAN_PAM_COL, "PAM")}

          {/* Shock / reward stimulus indicator */}
          {danPPL1 > 0.1 && (
            <text x={svgW / 2 - 80} y={danY - 30} textAnchor="middle" fontSize={10}
              fill={DAN_PPL1_COL} fontFamily="inherit" fontWeight="bold" opacity={danPPL1}>
              ⚡ SHOCK
            </text>
          )}
          {danPAM > 0.1 && (
            <text x={svgW / 2 + 80} y={danY - 30} textAnchor="middle" fontSize={10}
              fill={DAN_PAM_COL} fontFamily="inherit" fontWeight="bold" opacity={danPAM}>
              ✦ SUGAR
            </text>
          )}

          {/* KC circles */}
          {kcPositions.map((pos, i) => {
            const isA = ODOR_A.indices.includes(i);
            const isB = ODOR_B.indices.includes(i);
            const actA = isA ? kcGlowA : 0;
            const actB = isB ? kcGlowB : 0;
            if (actA > 0.01 && actB > 0.01) return cellGlow(pos.x, pos.y, 8, Math.max(actA, actB), "#fff", null, `kc-${i}`);
            if (actA > 0.01) return cellGlow(pos.x, pos.y, 8, actA, ODOR_A.color, null, `kc-${i}`);
            if (actB > 0.01) return cellGlow(pos.x, pos.y, 8, actB, ODOR_B.color, null, `kc-${i}`);
            return cellGlow(pos.x, pos.y, 8, 0, "#666", null, `kc-${i}`);
          })}

          {/* MBONs */}
          {approachPositions.map((pos, i) =>
            cellGlow(pos.x, pos.y, 16, approachFiring[i], APPROACH_COL, `A${i + 1}`, `ap-${i}`))}
          {avoidPositions.map((pos, i) =>
            cellGlow(pos.x, pos.y, 16, avoidFiring[i], AVOID_COL, `V${i + 1}`, `av-${i}`))}

          {/* Per-odor weight labels */}
          {(odorAOn || isTraining) && (
            <>
              <text x={approachX} y={mbonY + 76} textAnchor="middle" fontSize={8}
                fill={ODOR_A.color} fontFamily="inherit" opacity={0.6}>
                A→App: {(wAap * 100).toFixed(0)}%</text>
              <text x={avoidX} y={mbonY + 76} textAnchor="middle" fontSize={8}
                fill={ODOR_A.color} fontFamily="inherit" opacity={0.6}>
                A→Avd: {(wAav * 100).toFixed(0)}%</text>
            </>
          )}
          {(odorBOn || isTraining) && (
            <>
              <text x={approachX} y={mbonY + ((odorAOn || isTraining) ? 88 : 76)} textAnchor="middle" fontSize={8}
                fill={ODOR_B.color} fontFamily="inherit" opacity={0.6}>
                B→App: {(wBap * 100).toFixed(0)}%</text>
              <text x={avoidX} y={mbonY + ((odorAOn || isTraining) ? 88 : 76)} textAnchor="middle" fontSize={8}
                fill={ODOR_B.color} fontFamily="inherit" opacity={0.6}>
                B→Avd: {(wBav * 100).toFixed(0)}%</text>
            </>
          )}

          {/* Behavior readout */}
          <g transform={`translate(${svgW / 2}, ${svgH - 36})`}>
            <rect x={-approachMean * 140} y={-8} width={Math.max(0, approachMean * 140)} height={16}
              rx={3} fill={APPROACH_COL} opacity={0.6} />
            <rect x={0} y={-8} width={Math.max(0, avoidMean * 140)} height={16}
              rx={3} fill={AVOID_COL} opacity={0.6} />
            <line x1={0} y1={-14} x2={0} y2={14} stroke="#444" strokeWidth={1} />
            <text x={-150} y={4} textAnchor="end" fontSize={9} fill={APPROACH_COL} fontFamily="inherit">← APPROACH</text>
            <text x={150} y={4} textAnchor="start" fontSize={9} fill={AVOID_COL} fontFamily="inherit">AVOID →</text>
            {Math.abs(netDrive) > 0.03 && (() => {
              const ax = -netDrive * 200;
              const isAp = netDrive > 0;
              return (
                <g>
                  <line x1={0} y1={-20} x2={ax} y2={-20}
                    stroke={isAp ? APPROACH_COL : AVOID_COL} strokeWidth={3} opacity={0.8}
                    markerEnd={`url(#${isAp ? "netArrowG" : "netArrowR"})`} />
                  <text x={ax + (ax < 0 ? -10 : 10)} y={-17}
                    textAnchor={ax < 0 ? "end" : "start"} fontSize={8}
                    fill={isAp ? APPROACH_COL : AVOID_COL} fontFamily="inherit" fontWeight="bold">NET</text>
                </g>
              );
            })()}
            {!odorAOn && !odorBOn && !isTraining && (
              <text x={0} y={-18} textAnchor="middle" fontSize={9} fill="#555570" fontFamily="inherit" fontStyle="italic">
                no odor presented</text>
            )}
            <text x={0} y={28} textAnchor="middle" fontSize={9} fill="#555570" fontFamily="inherit" letterSpacing="0.1em">
              BEHAVIORAL DRIVE</text>
          </g>
        </svg>
      </div>

      {/* Weight bars */}
      <div style={{ display: "flex", gap: 20, marginTop: 14, justifyContent: "center", flexWrap: "wrap" }}>
        {[
          { label: "Odor A → Approach", weight: wAap, color: ODOR_A.color, active: odorAOn || (isTraining && stage.training === "aversive") },
          { label: "Odor A → Avoidance", weight: wAav, color: ODOR_A.color, active: odorAOn || isTraining },
          { label: "Odor B → Approach", weight: wBap, color: ODOR_B.color, active: odorBOn || isTraining },
          { label: "Odor B → Avoidance", weight: wBav, color: ODOR_B.color, active: odorBOn || (isTraining && stage.training === "appetitive") },
        ].map((item, i) => (
          <div key={i} style={{ textAlign: "center", width: 135, opacity: item.active ? 1 : 0.25, transition: "opacity 0.3s" }}>
            <div style={{ fontSize: 7, color: "#555570", letterSpacing: "0.08em", marginBottom: 3, textTransform: "uppercase" }}>
              {item.label}</div>
            <div style={{ height: 8, background: "#14141e", borderRadius: 4, overflow: "hidden", border: `1px solid ${BORDER}` }}>
              <div style={{
                height: "100%", width: `${item.weight * 100}%`,
                background: item.color, opacity: 0.7, borderRadius: 4,
                transition: "width 0.1s",
              }} />
            </div>
            <div style={{ fontSize: 11, color: item.color, fontWeight: 700, marginTop: 1 }}>
              {(item.weight * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div style={{
        marginTop: 16, padding: "12px 20px",
        background: PANEL, borderRadius: 8, border: `1px solid ${BORDER}`,
        maxWidth: 680, width: "90vw",
      }}>
        <div style={{ fontSize: 9, color: "#555570", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 6 }}>
          Circuit Elements
        </div>
        <div style={{ fontSize: 11, color: "#8888a0", lineHeight: 1.7 }}>
          <span style={{ color: ODOR_A.color }}>●</span> <strong style={{ color: "#bbb" }}>Odor A KCs</strong>{" "}
          <span style={{ color: ODOR_B.color }}>●</span> <strong style={{ color: "#bbb" }}>Odor B KCs</strong> — distinct sparse ensembles.{" "}
          <span style={{ color: APPROACH_COL }}>●</span> <strong style={{ color: "#bbb" }}>Approach MBONs</strong>{" "}
          <span style={{ color: AVOID_COL }}>●</span> <strong style={{ color: "#bbb" }}>Avoidance MBONs</strong> — opposing behavioral drives.{" "}
          <span style={{ color: DAN_PPL1_COL }}>●</span> <strong style={{ color: "#bbb" }}>PPL1 DANs</strong> (⚡ shock) depress approach synapses.{" "}
          <span style={{ color: DAN_PAM_COL }}>●</span> <strong style={{ color: "#bbb" }}>PAM DANs</strong> (✦ sugar) depress avoidance synapses.{" "}
          Watch the training phases to see DAN activation and real-time synaptic depression, then toggle odors to test the conditioned responses.
        </div>
      </div>
    </div>
  );
}
