import { useState, useCallback } from 'react'
import { Shield, AlertCircle, Activity, Cpu, Database, Zap, GitBranch } from 'lucide-react'
import UploadZone from './components/UploadZone'
import ResultPanel from './components/ResultPanel'
import { usePredict } from './hooks/usePredict'

const STATS = [
  { icon: Cpu,       label: 'Stage 1 model',  value: 'X3D-S',           sub: '3.8M params · ~10ms' },
  { icon: Zap,       label: 'Stage 2 model',  value: 'VideoMAE ViT-B',  sub: '86M params · ~200ms' },
  { icon: Database,  label: 'Dataset',        value: 'DCSASS',          sub: '16,639 clips' },
  { icon: GitBranch, label: 'Test accuracy',  value: '99.16%',          sub: '1,664 test samples' },
]

const CLASSES = [
  { name: 'Abuse',         sev: 'high'     },
  { name: 'Arrest',        sev: 'medium'   },
  { name: 'Arson',         sev: 'critical' },
  { name: 'Assault',       sev: 'high'     },
  { name: 'Burglary',      sev: 'medium'   },
  { name: 'Explosion',     sev: 'critical' },
  { name: 'Fighting',      sev: 'high'     },
  { name: 'RoadAccidents', sev: 'medium'   },
  { name: 'Robbery',       sev: 'high'     },
  { name: 'Shooting',      sev: 'critical' },
  { name: 'Shoplifting',   sev: 'low'      },
  { name: 'Stealing',      sev: 'low'      },
  { name: 'Vandalism',     sev: 'low'      },
]

const SEV_COLOR = { critical: '#EF4444', high: '#F97316', medium: '#F59E0B', low: '#22C55E' }

export default function App() {
  const [file, setFile] = useState(null)
  const { predict, result, loading, error, reset } = usePredict()

  const handleFile = useCallback((f) => { setFile(f); predict(f) }, [predict])
  const handleClear = useCallback(() => { setFile(null); reset() }, [reset])

  return (
    <div style={s.page}>
      <div style={s.topGlow} />

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header style={s.header}>
        <div style={s.logo}>
          <Shield size={20} color="var(--amber)" strokeWidth={1.5} />
          <span style={s.logoText}>VisionGuard</span>
        </div>
        <div style={s.headerRight}>
          <span style={s.statusDot} />
          <span style={s.statusText}>System online</span>
          <span style={s.headerTag}>X3D-S → VideoMAE · 99.16% acc</span>
        </div>
      </header>

      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <section style={s.hero}>
        <p style={s.eyebrow}>Crime classification · DCSASS · 13 categories</p>
        <h1 style={s.heroTitle}>
          Surveillance<br />
          <span style={s.heroAccent}>Crime Detection</span>
        </h1>
        <p style={s.heroSub}>
          Upload a surveillance clip. A two-stage pipeline first determines
          if the footage is normal or suspicious, then classifies the specific
          crime type — all in under 300ms.
        </p>
      </section>

      {/* ── Stats row ──────────────────────────────────────────────────── */}
      <div style={s.statsRow}>
        {STATS.map(({ icon: Icon, label, value, sub }) => (
          <div key={label} style={s.statCard}>
            <div style={s.statTop}>
              <Icon size={13} color="var(--amber)" strokeWidth={1.5} />
              <span style={s.statLabel}>{label}</span>
            </div>
            <p style={s.statValue}>{value}</p>
            <p style={s.statSub}>{sub}</p>
          </div>
        ))}
      </div>

      {/* ── Main panel ─────────────────────────────────────────────────── */}
      <main style={s.main}>
        <div style={s.card}>
          <div style={s.cardHeader}>
            <Activity size={14} color="var(--amber)" />
            <span style={s.cardTitle}>Upload footage</span>
          </div>
          <UploadZone
            onFile={handleFile}
            loading={loading}
            currentFile={file}
            onClear={handleClear}
          />
        </div>

        <div style={s.card}>
          <div style={s.cardHeader}>
            <Shield size={14} color="var(--amber)" />
            <span style={s.cardTitle}>Analysis results</span>
          </div>

          {error && (
            <div style={s.error} className="fade-up">
              <AlertCircle size={15} color="var(--red)" />
              <span>{error}</span>
            </div>
          )}

          {!result && !error && (
            <div style={s.empty}>
              <div style={s.emptyGrid}>
                {Array.from({ length: 13 }).map((_, i) => (
                  <div key={i} style={s.emptyBar}>
                    <div style={{ ...s.emptyFill, width: `${25 + (i * 17) % 55}%` }} />
                  </div>
                ))}
              </div>
              <p style={s.emptyText}>
                {loading ? 'Processing footage…' : 'Results will appear here after analysis'}
              </p>
            </div>
          )}

          {result && !error && <ResultPanel result={result} />}
        </div>
      </main>

      {/* ── How it works ───────────────────────────────────────────────── */}
      <section style={s.section}>
        <h2 style={s.sectionTitle}>How it works</h2>
        <div style={s.pipeline}>

          <div style={s.pipelineStep}>
            <div style={s.stepNum}>01</div>
            <div>
              <p style={s.stepTitle}>Upload a surveillance clip</p>
              <p style={s.stepDesc}>
                Drop any mp4, avi, mov, or mkv file up to 100 MB.
                The system samples 16 frames uniformly across the clip duration.
              </p>
            </div>
          </div>

          <div style={s.pipelineArrow}>→</div>

          <div style={s.pipelineStep}>
            <div style={{ ...s.stepNum, color: '#F97316', borderColor: '#F9731633' }}>02</div>
            <div>
              <p style={s.stepTitle}>Stage 1 — Binary gate (X3D-S)</p>
              <p style={s.stepDesc}>
                A lightweight X3D-S model runs in ~10ms and decides: normal or abnormal?
                Normal clips stop here. Only suspicious footage triggers Stage 2.
              </p>
            </div>
          </div>

          <div style={s.pipelineArrow}>→</div>

          <div style={s.pipelineStep}>
            <div style={{ ...s.stepNum, color: 'var(--amber)', borderColor: 'var(--amber-glow)' }}>03</div>
            <div>
              <p style={s.stepTitle}>Stage 2 — Crime classifier (VideoMAE)</p>
              <p style={s.stepDesc}>
                VideoMAE ViT-B (86M parameters, fine-tuned on DCSASS) classifies
                the specific crime type with confidence scores for all 13 categories.
              </p>
            </div>
          </div>

        </div>
      </section>

      {/* ── Crime categories ───────────────────────────────────────────── */}
      <section style={s.section}>
        <h2 style={s.sectionTitle}>Detectable crime categories</h2>
        <p style={s.sectionSub}>
          13 categories from the DCSASS surveillance dataset, colour-coded by severity
        </p>
        <div style={s.classGrid}>
          {CLASSES.map(({ name, sev }) => (
            <div key={name} style={s.classChip}>
              <span style={{ ...s.chipDot, background: SEV_COLOR[sev] }} />
              <span style={s.chipName}>{name}</span>
              <span style={{ ...s.chipSev, color: SEV_COLOR[sev] }}>{sev}</span>
            </div>
          ))}
        </div>

        <div style={s.legend}>
          {Object.entries(SEV_COLOR).map(([sev, color]) => (
            <div key={sev} style={s.legendItem}>
              <span style={{ ...s.chipDot, background: color }} />
              <span style={s.legendLabel}>{sev}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────────────── */}
      <footer style={s.footer}>
        <span>Built by Eyad Alatifi</span>
        <span style={s.footerDot} />
        <span>VideoMAE ViT-B · Fine-tuned on DCSASS</span>
        <span style={s.footerDot} />
        <a href="https://github.com/00ed" target="_blank" rel="noreferrer" style={s.footerLink}>
          GitHub
        </a>
      </footer>
    </div>
  )
}

const s = {
  page: {
    minHeight: '100vh', display: 'flex', flexDirection: 'column',
    maxWidth: '1100px', margin: '0 auto', padding: '0 24px 64px',
    position: 'relative',
  },
  topGlow: {
    position: 'fixed', top: 0, left: '50%', transform: 'translateX(-50%)',
    width: '600px', height: '1px',
    background: 'linear-gradient(90deg, transparent, var(--amber), transparent)',
    zIndex: 10,
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '20px 0', borderBottom: '1px solid var(--border)',
  },
  logo: { display: 'flex', alignItems: 'center', gap: '10px' },
  logoText: {
    fontFamily: 'var(--display)', fontWeight: 700, fontSize: '17px',
    color: 'var(--text-1)', letterSpacing: '-0.01em',
  },
  headerRight: { display: 'flex', alignItems: 'center', gap: '10px' },
  statusDot: {
    display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%',
    background: 'var(--green)', animation: 'pulse-amber 2.5s ease-in-out infinite',
  },
  statusText: { fontSize: '12px', color: 'var(--text-2)', fontFamily: 'var(--mono)' },
  headerTag: {
    fontSize: '11px', color: 'var(--text-3)', fontFamily: 'var(--mono)',
    padding: '3px 8px', border: '1px solid var(--border)',
    borderRadius: '4px', background: 'var(--surface)',
  },
  hero: { padding: '52px 0 36px', maxWidth: '560px' },
  eyebrow: {
    fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--amber)',
    textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: '16px',
  },
  heroTitle: {
    fontFamily: 'var(--display)', fontSize: 'clamp(36px, 5vw, 52px)',
    fontWeight: 700, lineHeight: 1.05, letterSpacing: '-0.03em',
    color: 'var(--text-1)', marginBottom: '18px',
  },
  heroAccent: { color: 'var(--amber)' },
  heroSub: { fontSize: '15px', color: 'var(--text-2)', lineHeight: 1.65, maxWidth: '460px' },

  // Stats
  statsRow: {
    display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '12px', marginBottom: '28px',
  },
  statCard: {
    background: 'var(--surface)', border: '1px solid var(--border)',
    borderRadius: '10px', padding: '14px 16px',
    display: 'flex', flexDirection: 'column', gap: '4px',
  },
  statTop: { display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' },
  statLabel: { fontSize: '11px', color: 'var(--text-3)', fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em' },
  statValue: { fontFamily: 'var(--display)', fontWeight: 700, fontSize: '17px', color: 'var(--text-1)' },
  statSub: { fontSize: '11px', color: 'var(--text-3)', fontFamily: 'var(--mono)' },

  // Main
  main: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', alignItems: 'start', marginBottom: '48px' },
  card: {
    background: 'var(--surface)', border: '1px solid var(--border)',
    borderRadius: '12px', padding: '20px',
    display: 'flex', flexDirection: 'column', gap: '16px',
  },
  cardHeader: {
    display: 'flex', alignItems: 'center', gap: '8px',
    paddingBottom: '14px', borderBottom: '1px solid var(--border)',
  },
  cardTitle: { fontFamily: 'var(--mono)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-2)' },
  error: {
    display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '14px',
    background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)',
    borderRadius: '8px', fontSize: '13px', color: 'var(--red)', lineHeight: 1.5,
  },
  empty: { display: 'flex', flexDirection: 'column', gap: '20px', padding: '8px 0' },
  emptyGrid: { display: 'flex', flexDirection: 'column', gap: '8px', opacity: 0.2 },
  emptyBar: { height: '6px', background: 'var(--surface-2)', borderRadius: '3px', overflow: 'hidden' },
  emptyFill: { height: '100%', background: 'var(--border-hover)', borderRadius: '3px' },
  emptyText: { fontSize: '13px', color: 'var(--text-3)', fontFamily: 'var(--mono)', textAlign: 'center' },

  // How it works
  section: { marginBottom: '48px' },
  sectionTitle: {
    fontFamily: 'var(--display)', fontSize: '22px', fontWeight: 700,
    color: 'var(--text-1)', letterSpacing: '-0.02em', marginBottom: '8px',
  },
  sectionSub: { fontSize: '13px', color: 'var(--text-3)', marginBottom: '20px' },
  pipeline: {
    display: 'flex', alignItems: 'flex-start', gap: '8px',
    background: 'var(--surface)', border: '1px solid var(--border)',
    borderRadius: '12px', padding: '24px',
  },
  pipelineStep: { flex: 1, display: 'flex', flexDirection: 'column', gap: '10px' },
  pipelineArrow: {
    color: 'var(--text-3)', fontSize: '18px', paddingTop: '6px',
    flexShrink: 0, alignSelf: 'flex-start', marginTop: '4px',
  },
  stepNum: {
    fontFamily: 'var(--mono)', fontSize: '11px', fontWeight: 700,
    color: '#22C55E', border: '1px solid rgba(34,197,94,0.2)',
    borderRadius: '4px', padding: '2px 7px', alignSelf: 'flex-start',
    letterSpacing: '0.06em',
  },
  stepTitle: { fontSize: '13px', fontWeight: 600, color: 'var(--text-1)', marginBottom: '4px' },
  stepDesc: { fontSize: '12px', color: 'var(--text-3)', lineHeight: 1.6 },

  // Classes
  classGrid: {
    display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
    gap: '8px', marginBottom: '16px',
  },
  classChip: {
    display: 'flex', alignItems: 'center', gap: '8px',
    background: 'var(--surface)', border: '1px solid var(--border)',
    borderRadius: '8px', padding: '9px 12px',
  },
  chipDot: { width: '7px', height: '7px', borderRadius: '50%', flexShrink: 0 },
  chipName: { flex: 1, fontSize: '13px', color: 'var(--text-1)' },
  chipSev: { fontSize: '10px', fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.06em', flexShrink: 0 },
  legend: { display: 'flex', gap: '20px', flexWrap: 'wrap' },
  legendItem: { display: 'flex', alignItems: 'center', gap: '6px' },
  legendLabel: { fontSize: '12px', color: 'var(--text-3)', fontFamily: 'var(--mono)' },

  // Footer
  footer: {
    marginTop: 'auto', paddingTop: '32px', borderTop: '1px solid var(--border)',
    display: 'flex', alignItems: 'center', gap: '10px',
    fontSize: '12px', color: 'var(--text-3)', fontFamily: 'var(--mono)',
  },
  footerDot: { width: '3px', height: '3px', borderRadius: '50%', background: 'var(--text-3)' },
  footerLink: { color: 'var(--amber)', textDecoration: 'none' },
}
