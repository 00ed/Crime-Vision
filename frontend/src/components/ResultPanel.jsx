import { ShieldAlert, ShieldCheck, Clock, Zap } from 'lucide-react'

const SEVERITY = {
  Explosion: 'critical', Shooting: 'critical', Arson: 'critical',
  Assault: 'high',  Fighting: 'high',  Robbery: 'high',  Abuse: 'high',
  Arrest: 'medium', RoadAccidents: 'medium', Burglary: 'medium',
  Vandalism: 'low', Shoplifting: 'low', Stealing: 'low',
}

const SEVERITY_COLOR = {
  critical: '#EF4444',
  high:     '#F97316',
  medium:   '#F59E0B',
  low:      '#22C55E',
}

const formatPercent = (value) => (
  Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '--'
)

const formatMs = (value) => (
  Number.isFinite(value) ? `${value.toFixed(0)} ms` : '--'
)

export default function ResultPanel({ result }) {
  const {
    is_normal, stage1_confidence,
    top_prediction, confidence, all_scores,
    inference_ms, stage1_ms, stage2_ms,
  } = result

  // ── Normal clip ────────────────────────────────────────────────────────────
  if (is_normal) {
    return (
      <div style={styles.wrapper} className="fade-up">
        <div style={styles.normalBanner}>
          <ShieldCheck size={22} color="var(--green)" strokeWidth={1.5} />
          <div>
            <p style={styles.bannerLabel}>Stage 1 verdict</p>
            <p style={{ ...styles.bannerPrediction, color: 'var(--green)' }}>
              Normal activity
            </p>
          </div>
          <p style={{ ...styles.confidence, color: 'var(--green)', marginLeft: 'auto' }}>
            {formatPercent(stage1_confidence)}
          </p>
        </div>

        <div style={styles.stageRow}>
          <StageTag label="Stage 1" value={formatMs(stage1_ms)} active />
          <StageTag label="Stage 2" value="skipped" dimmed />
        </div>

        <p style={styles.normalNote}>
          The binary classifier determined this clip contains no anomalous activity.
          Stage 2 was not triggered.
        </p>
      </div>
    )
  }

  // ── Abnormal clip ──────────────────────────────────────────────────────────
  const severity      = SEVERITY[top_prediction] || 'medium'
  const severityColor = SEVERITY_COLOR[severity]

  return (
    <div style={styles.wrapper} className="fade-up">

      {/* Stage 1 badge */}
      <div style={styles.stageRow}>
        <StageTag label="Stage 1" value={`Abnormal · ${stage1_ms?.toFixed(0)} ms`} active warn />
        <StageTag label="Stage 2" value={`${stage2_ms?.toFixed(0)} ms`} active />
      </div>

      {/* Top prediction banner */}
      <div style={{ ...styles.banner, borderColor: severityColor + '44', background: severityColor + '0D' }}>
        <div style={styles.bannerLeft}>
          <ShieldAlert size={18} color={severityColor} strokeWidth={1.5} />
          <div>
            <p style={styles.bannerLabel}>Detected activity</p>
            <p style={{ ...styles.bannerPrediction, color: severityColor }}>
              {top_prediction}
            </p>
          </div>
        </div>
        <div style={styles.bannerRight}>
          <p style={{ ...styles.confidence, color: severityColor }}>
            {formatPercent(confidence)}
          </p>
          <p style={styles.severityTag}>
            <span style={{ ...styles.severityDot, background: severityColor }} />
            {severity}
          </p>
        </div>
      </div>

      {/* Timing */}
      <div style={styles.meta}>
        <Clock size={12} color="var(--text-3)" />
        <span style={styles.metaText}>
          Total: <span style={styles.metaMono}>{formatMs(inference_ms)}</span>
        </span>
        <span style={styles.metaDivider} />
        <span style={styles.metaText}>
          Model: <span style={styles.metaMono}>X3D-S → VideoMAE ViT-B</span>
        </span>
      </div>

      {/* All class scores */}
      {Array.isArray(all_scores) && all_scores.length > 0 && (
        <div style={styles.scoresSection}>
          <p style={styles.sectionLabel}>All class scores</p>
          <div style={styles.scoresList}>
            {all_scores.map((item, i) => {
              const isTop    = item.label === top_prediction
              const sev      = SEVERITY[item.label] || 'medium'
              const color    = isTop ? severityColor : 'var(--text-3)'
              const itemConfidence = Number.isFinite(item.confidence) ? item.confidence : 0
              const pct      = (itemConfidence * 100).toFixed(1)

              return (
                <div key={item.label} style={styles.scoreRow}>
                  <span style={{
                    ...styles.scoreLabel,
                    color:      isTop ? 'var(--text-1)' : 'var(--text-2)',
                    fontWeight: isTop ? 600 : 400,
                  }}>
                    {item.label}
                  </span>
                  <div style={styles.barTrack}>
                    <div style={{
                      ...styles.barFill,
                      width: `${itemConfidence * 100}%`,
                      background: isTop
                        ? `linear-gradient(90deg, ${severityColor}88, ${severityColor})`
                        : 'var(--surface-2)',
                      animationDelay: `${i * 0.03}s`,
                    }} />
                  </div>
                  <span style={{ ...styles.scorePct, color }}>
                    {pct}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function StageTag({ label, value, active, dimmed, warn }) {
  const color = dimmed
    ? 'var(--text-3)'
    : warn
    ? '#F97316'
    : 'var(--amber)'

  return (
    <div style={{
      ...stageBadge,
      borderColor: active ? color + '44' : 'var(--border)',
      background:  active ? color + '0D' : 'transparent',
      opacity:     dimmed ? 0.5 : 1,
    }}>
      <Zap size={11} color={color} />
      <span style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--text-2)' }}>
        {label}
      </span>
      <span style={{ fontFamily: 'var(--mono)', fontSize: '11px', color }}>
        {value}
      </span>
    </div>
  )
}

const stageBadge = {
  display:      'flex',
  alignItems:   'center',
  gap:          '6px',
  padding:      '5px 10px',
  borderRadius: '6px',
  border:       '1px solid',
}

const styles = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: '14px' },
  stageRow: { display: 'flex', gap: '8px', flexWrap: 'wrap' },
  normalBanner: {
    display: 'flex', alignItems: 'center', gap: '14px',
    padding: '16px 20px', borderRadius: '10px',
    border: '1px solid rgba(34,197,94,0.25)',
    background: 'rgba(34,197,94,0.06)',
  },
  banner: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '16px 20px', borderRadius: '10px', border: '1px solid',
  },
  bannerLeft:  { display: 'flex', alignItems: 'center', gap: '12px' },
  bannerRight: { textAlign: 'right' },
  bannerLabel: {
    fontSize: '11px', color: 'var(--text-3)',
    fontFamily: 'var(--mono)', textTransform: 'uppercase',
    letterSpacing: '0.08em', marginBottom: '2px',
  },
  bannerPrediction: {
    fontFamily: 'var(--display)', fontSize: '22px',
    fontWeight: 700, lineHeight: 1.1,
  },
  confidence: {
    fontFamily: 'var(--mono)', fontSize: '26px',
    fontWeight: 500, lineHeight: 1,
  },
  severityTag: {
    display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
    gap: '5px', fontSize: '11px', color: 'var(--text-2)',
    fontFamily: 'var(--mono)', textTransform: 'uppercase',
    letterSpacing: '0.08em', marginTop: '4px',
  },
  severityDot: { width: '6px', height: '6px', borderRadius: '50%', flexShrink: 0 },
  meta: { display: 'flex', alignItems: 'center', gap: '8px', padding: '0 2px' },
  metaText:    { fontSize: '12px', color: 'var(--text-3)' },
  metaMono:    { fontFamily: 'var(--mono)', color: 'var(--text-2)' },
  metaDivider: { width: '1px', height: '12px', background: 'var(--border)' },
  normalNote: {
    fontSize: '13px', color: 'var(--text-3)',
    fontFamily: 'var(--mono)', lineHeight: 1.6,
    padding: '12px', background: 'var(--surface-2)',
    borderRadius: '8px', border: '1px solid var(--border)',
  },
  scoresSection: {
    background: 'var(--surface)', border: '1px solid var(--border)',
    borderRadius: '10px', padding: '16px 20px',
    display: 'flex', flexDirection: 'column', gap: '12px',
  },
  sectionLabel: {
    fontSize: '11px', fontFamily: 'var(--mono)',
    color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em',
  },
  scoresList: { display: 'flex', flexDirection: 'column', gap: '8px' },
  scoreRow: {
    display: 'grid', gridTemplateColumns: '140px 1fr 52px',
    alignItems: 'center', gap: '10px',
  },
  scoreLabel: {
    fontSize: '13px', fontFamily: 'var(--sans)',
    overflow: 'hidden', whiteSpace: 'nowrap',
  },
  barTrack: {
    height: '6px', background: 'var(--surface-2)',
    borderRadius: '3px', overflow: 'hidden',
  },
  barFill: {
    height: '100%', borderRadius: '3px', minWidth: '2px',
    animation: 'bar-fill 0.6s ease both',
  },
  scorePct: { fontFamily: 'var(--mono)', fontSize: '12px', textAlign: 'right' },
}
