import React, { useCallback, useEffect, useRef, useState } from 'react'
import ReactDOM from 'react-dom/client'
import axios from 'axios'
import {
  Activity,
  AlertCircle,
  ChevronUp,
  Clock,
  Film,
  Shield,
  ShieldAlert,
  Upload,
  X,
} from 'lucide-react'

const ACCEPTED_VIDEO_TYPES = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska']
const EMPTY_BAR_WIDTHS = [64, 42, 58, 35, 72, 48, 55, 30, 68, 44, 52, 38, 60]

const SEVERITY = {
  Explosion: 'critical',
  Shooting: 'critical',
  Arson: 'critical',
  Assault: 'high',
  Fighting: 'high',
  Robbery: 'high',
  Abuse: 'high',
  Arrest: 'medium',
  RoadAccidents: 'medium',
  Burglary: 'medium',
  Vandalism: 'low',
  Shoplifting: 'low',
  Stealing: 'low',
}

const SEVERITY_COLOR = {
  critical: '#EF4444',
  high: '#F97316',
  medium: '#F59E0B',
  low: '#22C55E',
}

function GlobalStyles() {
  return (
    <style>{`
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

      :root {
        --bg: #0B0E0E;
        --surface: #131818;
        --surface-2: #1A2020;
        --border: #1E2828;
        --border-hover: #2A3838;
        --amber: #F59E0B;
        --amber-dim: #B45309;
        --amber-glow: rgba(245, 158, 11, 0.12);
        --amber-glow-s: rgba(245, 158, 11, 0.06);
        --red: #EF4444;
        --green: #22C55E;
        --text-1: #F1F5F9;
        --text-2: #94A3B8;
        --text-3: #4B5E6B;
        --mono: 'JetBrains Mono', monospace;
        --sans: 'Inter', sans-serif;
        --display: 'Space Grotesk', sans-serif;
      }

      html { scroll-behavior: smooth; }

      body {
        background: var(--bg);
        color: var(--text-1);
        font-family: var(--sans);
        font-size: 15px;
        line-height: 1.6;
        min-height: 100vh;
        -webkit-font-smoothing: antialiased;
      }

      button, input { font: inherit; }
      ::-webkit-scrollbar { width: 6px; }
      ::-webkit-scrollbar-track { background: var(--bg); }
      ::-webkit-scrollbar-thumb { background: var(--border-hover); border-radius: 3px; }

      @keyframes scan {
        0% { top: 0%; opacity: 1; }
        100% { top: 100%; opacity: 0; }
      }

      @keyframes pulse-amber {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
      }

      @keyframes fade-up {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
      }

      @keyframes bar-fill {
        from { width: 0%; }
      }

      @keyframes spin {
        to { transform: rotate(360deg); }
      }

      .fade-up { animation: fade-up 0.4s ease both; }
      .fade-up-1 { animation: fade-up 0.4s ease both; }
      .fade-up-2 { animation: fade-up 0.4s 0.08s ease both; }
      .fade-up-3 { animation: fade-up 0.4s 0.16s ease both; }

      @media (max-width: 780px) {
        .app-header,
        .app-footer,
        .result-banner,
        .result-meta {
          align-items: flex-start !important;
          flex-direction: column !important;
        }

        .app-main {
          grid-template-columns: 1fr !important;
        }

        .score-row {
          grid-template-columns: 100px 1fr 48px !important;
        }
      }
    `}</style>
  )
}

function usePredict() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const predict = useCallback(async (file) => {
    setLoading(true)
    setResult(null)
    setError(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const { data } = await axios.post('/api/predict', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(data)
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        'Prediction failed. Is the backend running?'
      setError(message)
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  return { predict, result, loading, error, reset }
}

function UploadZone({ onFile, loading, currentFile, onClear }) {
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)

  const handleFile = useCallback((file) => {
    if (!file) return

    const hasValidType = ACCEPTED_VIDEO_TYPES.includes(file.type)
    const hasValidExtension = /\.(mp4|avi|mov|mkv)$/i.test(file.name)

    if (!hasValidType && !hasValidExtension) {
      alert('Unsupported format. Please upload mp4, avi, mov, or mkv.')
      return
    }

    if (file.size > 100 * 1024 * 1024) {
      alert('File too large. Max 100 MB.')
      return
    }

    onFile(file)
  }, [onFile])

  useEffect(() => {
    if (!currentFile) {
      setPreviewUrl(null)
      return undefined
    }

    const url = URL.createObjectURL(currentFile)
    setPreviewUrl(url)

    return () => URL.revokeObjectURL(url)
  }, [currentFile])

  const onDrop = (event) => {
    event.preventDefault()
    setDrag(false)
    handleFile(event.dataTransfer.files[0])
  }

  return (
    <div style={uploadStyles.wrapper}>
      <div
        style={{
          ...uploadStyles.zone,
          ...(drag ? uploadStyles.zoneDrag : {}),
          ...(loading ? uploadStyles.zoneLoading : {}),
        }}
        onDragOver={(event) => {
          event.preventDefault()
          setDrag(true)
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        onClick={() => !currentFile && !loading && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(event) => handleFile(event.target.files[0])}
        />

        {loading && <div style={uploadStyles.scanLine} />}

        {currentFile ? (
          <div style={uploadStyles.preview}>
            <video
              src={previewUrl}
              style={uploadStyles.video}
              muted
              loop
              autoPlay
              playsInline
            />
            <div style={uploadStyles.previewOverlay}>
              <Film size={14} color="var(--amber)" />
              <span style={uploadStyles.filename}>{currentFile.name}</span>
              <span style={uploadStyles.filesize}>
                {(currentFile.size / (1024 * 1024)).toFixed(1)} MB
              </span>
            </div>
            {!loading && (
              <>
                <button
                  type="button"
                  style={uploadStyles.clearBtn}
                  onClick={(event) => {
                    event.stopPropagation()
                    onClear()
                  }}
                  title="Remove video"
                >
                  <X size={14} />
                </button>
              </>
            )}
          </div>
        ) : (
          <div style={uploadStyles.emptyState}>
            <div style={uploadStyles.iconRing}>
              <Upload size={22} color="var(--amber)" strokeWidth={1.5} />
            </div>
            <p style={uploadStyles.headline}>Drop a surveillance clip here</p>
            <p style={uploadStyles.sub}>or click to browse - mp4, avi, mov, mkv - max 100 MB</p>
          </div>
        )}
      </div>

      {currentFile && !loading && (
        <button
          type="button"
          style={uploadStyles.analyseBtn}
          onClick={() => onFile(currentFile)}
          className="fade-up"
        >
          <span style={uploadStyles.btnDot} />
          Analyse clip
        </button>
      )}

      {loading && (
        <div style={uploadStyles.analysing} className="fade-up">
          <span style={uploadStyles.spinner} />
          Analysing footage...
        </div>
      )}
    </div>
  )
}

function ResultPanel({ result }) {
  const {
    is_normal: isNormal,
    stage1_confidence: stage1Confidence,
    top_prediction: topPrediction,
    confidence,
    all_scores: allScores,
    inference_ms: inferenceMs,
    stage1_ms: stage1Ms,
    stage2_ms: stage2Ms,
  } = result

  const displayPrediction = isNormal ? 'Normal activity' : (topPrediction || 'Unknown activity')
  const displayConfidence = isNormal ? stage1Confidence : confidence
  const scores = Array.isArray(allScores) ? allScores : []
  const severity = isNormal ? 'low' : (SEVERITY[displayPrediction] || 'medium')
  const severityColor = isNormal ? '#22C55E' : SEVERITY_COLOR[severity]
  const confidenceText = Number.isFinite(displayConfidence)
    ? `${(displayConfidence * 100).toFixed(1)}%`
    : '--'
  const totalMs = Number.isFinite(inferenceMs) ? `${inferenceMs.toFixed(0)} ms` : '--'
  const stage1Text = Number.isFinite(stage1Ms) ? `${stage1Ms.toFixed(0)} ms` : '--'
  const stage2Text = isNormal
    ? 'skipped'
    : Number.isFinite(stage2Ms)
      ? `${stage2Ms.toFixed(0)} ms`
      : '--'

  return (
    <div style={resultStyles.wrapper} className="fade-up">
      <div
        className="result-banner"
        style={{
          ...resultStyles.banner,
          borderColor: `${severityColor}44`,
          background: `${severityColor}0D`,
        }}
      >
        <div style={resultStyles.bannerLeft}>
          <ShieldAlert size={18} color={severityColor} strokeWidth={1.5} />
          <div>
            <p style={resultStyles.bannerLabel}>Detected activity</p>
            <p style={{ ...resultStyles.bannerPrediction, color: severityColor }}>
              {displayPrediction}
            </p>
          </div>
        </div>
        <div style={resultStyles.bannerRight}>
          <p style={{ ...resultStyles.confidence, color: severityColor }}>
            {confidenceText}
          </p>
          <p style={resultStyles.severityTag}>
            <span style={{ ...resultStyles.severityDot, background: severityColor }} />
            {isNormal ? 'normal' : severity}
          </p>
        </div>
      </div>

      <div className="result-meta" style={resultStyles.meta}>
        <Clock size={12} color="var(--text-3)" />
        <span style={resultStyles.metaText}>
          Inference: <span style={resultStyles.metaMono}>{totalMs}</span>
        </span>
        <span style={resultStyles.metaDivider} />
        <span style={resultStyles.metaText}>
          Stage 1: <span style={resultStyles.metaMono}>{stage1Text}</span>
        </span>
        <span style={resultStyles.metaDivider} />
        <span style={resultStyles.metaText}>
          Stage 2: <span style={resultStyles.metaMono}>{stage2Text}</span>
        </span>
      </div>

      <div style={resultStyles.summary}>
        <p style={resultStyles.summaryText}>
          {isNormal
            ? 'This clip was classified as normal and stage 2 was skipped. Use the score panel to compare other class confidences.'
            : 'Suspicious activity was detected. Review the top prediction and confidence chart below to understand the model output.'}
        </p>
      </div>

      {scores.length > 0 && (
        <div style={resultStyles.scoresSection}>
          <p style={resultStyles.sectionLabel}>All class scores</p>
          <div style={resultStyles.scoresList}>
            {scores.map((item, index) => {
              const itemConfidence = Number.isFinite(item.confidence) ? item.confidence : 0
              const isTop = item.label === displayPrediction
              const color = isTop ? severityColor : 'var(--text-3)'
              const percent = (itemConfidence * 100).toFixed(1)

              return (
                <div
                  key={item.label}
                  style={resultStyles.scoreRow}
                  className={`score-row ${index < 3 ? `fade-up-${index + 1}` : ''}`}
                >
                  <div style={resultStyles.scoreLabel}>
                    {isTop && <ChevronUp size={11} color={severityColor} />}
                    <span style={{ color: isTop ? 'var(--text-1)' : 'var(--text-2)', fontWeight: isTop ? 600 : 400 }}>
                      {item.label}
                    </span>
                  </div>

                  <div style={resultStyles.barTrack}>
                    <div
                      style={{
                        ...resultStyles.barFill,
                        width: `${itemConfidence * 100}%`,
                        background: isTop
                          ? `linear-gradient(90deg, ${severityColor}88, ${severityColor})`
                          : 'var(--surface-2)',
                        animation: 'bar-fill 0.6s ease both',
                        animationDelay: `${index * 0.03}s`,
                      }}
                    />
                  </div>

                  <span style={{ ...resultStyles.scorePct, color }}>
                    {percent}%
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

function App() {
  const [file, setFile] = useState(null)
  const { predict, result, loading, error, reset } = usePredict()

  const handleFile = useCallback((selectedFile) => {
    setFile(selectedFile)
    predict(selectedFile)
  }, [predict])

  const handleClear = useCallback(() => {
    setFile(null)
    reset()
  }, [reset])

  return (
    <>
      <GlobalStyles />
      <div style={appStyles.page}>
        <div style={appStyles.topGlow} />

        <header className="app-header" style={appStyles.header}>
          <div style={appStyles.logo}>
            <Shield size={20} color="var(--amber)" strokeWidth={1.5} />
            <span style={appStyles.logoText}>VisionGuard</span>
          </div>
          <div style={appStyles.headerRight}>
            <span style={appStyles.statusDot} />
            <span style={appStyles.statusText}>System online</span>
            <span style={appStyles.headerTag}>VideoMAE ViT-B - 99.16% acc</span>
          </div>
        </header>

        <section style={appStyles.hero}>
          <p style={appStyles.eyebrow}>Crime classification - DCSASS - 13 categories</p>
          <h1 style={appStyles.heroTitle}>
            Surveillance<br />
            <span style={appStyles.heroAccent}>Crime Detection</span>
          </h1>
          <p style={appStyles.heroSub}>
            Upload a surveillance clip. VideoMAE analyses the footage and classifies
            the activity across 13 crime categories in seconds.
          </p>
        </section>

        <main className="app-main" style={appStyles.main}>
          <div style={appStyles.card}>
            <div style={appStyles.cardHeader}>
              <Activity size={14} color="var(--amber)" />
              <span style={appStyles.cardTitle}>Upload footage</span>
            </div>
            <UploadZone
              onFile={handleFile}
              loading={loading}
              currentFile={file}
              onClear={handleClear}
            />
          </div>

          <div style={appStyles.card}>
            <div style={appStyles.cardHeader}>
              <Shield size={14} color="var(--amber)" />
              <span style={appStyles.cardTitle}>Analysis results</span>
            </div>

            {error && (
              <div style={appStyles.error} className="fade-up">
                <AlertCircle size={15} color="var(--red)" />
                <span>{error}</span>
              </div>
            )}

            {!result && !error && (
              <div style={appStyles.empty}>
                <div style={appStyles.emptyGrid}>
                  {EMPTY_BAR_WIDTHS.map((width, index) => (
                    <div key={index} style={appStyles.emptyBar}>
                      <div style={{ ...appStyles.emptyBarFill, width: `${width}%` }} />
                    </div>
                  ))}
                </div>
                <p style={appStyles.emptyText}>
                  {loading
                    ? 'Processing footage...'
                    : 'Choose a clip and click Analyse clip to see classification and confidence scores.'}
                </p>
              </div>
            )}

            {result && !error && <ResultPanel result={result} />}
          </div>
        </main>

        <footer className="app-footer" style={appStyles.footer}>
          <span>Built by Eyad Alatifi</span>
          <span style={appStyles.footerDot} />
          <span>VideoMAE ViT-B - Fine-tuned on DCSASS</span>
          <span style={appStyles.footerDot} />
          <a
            href="https://github.com/00ed"
            target="_blank"
            rel="noreferrer"
            style={appStyles.footerLink}
          >
            GitHub
          </a>
        </footer>
      </div>
    </>
  )
}

const appStyles = {
  page: { minHeight: '100vh', display: 'flex', flexDirection: 'column', maxWidth: '1100px', margin: '0 auto', padding: '0 24px 48px', position: 'relative' },
  topGlow: { position: 'fixed', top: 0, left: '50%', transform: 'translateX(-50%)', width: '600px', height: '1px', background: 'linear-gradient(90deg, transparent, var(--amber), transparent)', zIndex: 10 },
  header: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', padding: '20px 0', borderBottom: '1px solid var(--border)' },
  logo: { display: 'flex', alignItems: 'center', gap: '10px' },
  logoText: { fontFamily: 'var(--display)', fontWeight: 700, fontSize: '17px', color: 'var(--text-1)' },
  headerRight: { display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' },
  statusDot: { display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%', background: 'var(--green)', animation: 'pulse-amber 2.5s ease-in-out infinite' },
  statusText: { fontSize: '12px', color: 'var(--text-2)', fontFamily: 'var(--mono)' },
  headerTag: { fontSize: '11px', color: 'var(--text-3)', fontFamily: 'var(--mono)', padding: '3px 8px', border: '1px solid var(--border)', borderRadius: '4px', background: 'var(--surface)' },
  hero: { padding: '52px 0 44px', maxWidth: '560px' },
  eyebrow: { fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--amber)', textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: '16px' },
  heroTitle: { fontFamily: 'var(--display)', fontSize: 'clamp(36px, 5vw, 52px)', fontWeight: 700, lineHeight: 1.05, color: 'var(--text-1)', marginBottom: '18px' },
  heroAccent: { color: 'var(--amber)' },
  heroSub: { fontSize: '15px', color: 'var(--text-2)', lineHeight: 1.65, maxWidth: '440px' },
  main: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', alignItems: 'start' },
  card: { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' },
  cardHeader: { display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '14px', borderBottom: '1px solid var(--border)' },
  cardTitle: { fontFamily: 'var(--mono)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-2)' },
  error: { display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '14px', background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: '8px', fontSize: '13px', color: 'var(--red)', lineHeight: 1.5 },
  empty: { display: 'flex', flexDirection: 'column', gap: '20px', padding: '8px 0' },
  emptyGrid: { display: 'flex', flexDirection: 'column', gap: '8px', opacity: 0.25 },
  emptyBar: { height: '6px', background: 'var(--surface-2)', borderRadius: '3px', overflow: 'hidden' },
  emptyBarFill: { height: '100%', background: 'var(--border-hover)', borderRadius: '3px' },
  emptyText: { fontSize: '13px', color: 'var(--text-3)', fontFamily: 'var(--mono)', textAlign: 'center' },
  footer: { marginTop: 'auto', paddingTop: '40px', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '12px', color: 'var(--text-3)', fontFamily: 'var(--mono)', flexWrap: 'wrap' },
  footerDot: { width: '3px', height: '3px', borderRadius: '50%', background: 'var(--text-3)' },
  footerLink: { color: 'var(--amber)', textDecoration: 'none' },
}

const uploadStyles = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: '14px' },
  zone: { position: 'relative', border: '1.5px dashed var(--border-hover)', borderRadius: '8px', background: 'var(--surface)', minHeight: '260px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', overflow: 'hidden', transition: 'border-color 0.2s, background 0.2s' },
  zoneDrag: { borderColor: 'var(--amber)', background: 'var(--amber-glow-s)' },
  zoneLoading: { cursor: 'default', borderColor: 'var(--amber-dim)' },
  scanLine: { position: 'absolute', left: 0, right: 0, height: '2px', background: 'linear-gradient(90deg, transparent, var(--amber), transparent)', animation: 'scan 1.8s linear infinite', zIndex: 10, pointerEvents: 'none' },
  preview: { width: '100%', height: '100%', position: 'relative' },
  video: { width: '100%', height: '240px', objectFit: 'cover', display: 'block', opacity: 0.7 },
  previewOverlay: { position: 'absolute', bottom: 0, left: 0, right: 0, padding: '8px 12px', background: 'linear-gradient(transparent, rgba(0,0,0,0.8))', display: 'flex', alignItems: 'center', gap: '8px', fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--text-2)' },
  filename: { flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-1)' },
  filesize: { color: 'var(--text-3)', flexShrink: 0 },
  clearBtn: { position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.6)', border: '1px solid var(--border-hover)', borderRadius: '6px', color: 'var(--text-2)', cursor: 'pointer', width: '28px', height: '28px', display: 'flex', alignItems: 'center', justifyContent: 'center' },
  emptyState: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px', padding: '40px 20px', userSelect: 'none' },
  iconRing: { width: '56px', height: '56px', borderRadius: '50%', border: '1.5px solid var(--border-hover)', background: 'var(--amber-glow-s)', display: 'flex', alignItems: 'center', justifyContent: 'center' },
  headline: { fontFamily: 'var(--display)', fontSize: '16px', fontWeight: 600, color: 'var(--text-1)' },
  sub: { fontSize: '13px', color: 'var(--text-3)', textAlign: 'center', maxWidth: '300px' },
  analyseBtn: { display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', width: '100%', padding: '13px', background: 'var(--amber)', color: '#0B0E0E', fontFamily: 'var(--display)', fontWeight: 700, fontSize: '15px', border: 'none', borderRadius: '8px', cursor: 'pointer' },
  btnDot: { width: '7px', height: '7px', borderRadius: '50%', background: '#0B0E0E', opacity: 0.6 },
  analysing: { display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', color: 'var(--amber)', fontFamily: 'var(--mono)', fontSize: '13px', padding: '10px' },
  spinner: { display: 'inline-block', width: '14px', height: '14px', border: '2px solid var(--amber-glow)', borderTopColor: 'var(--amber)', borderRadius: '50%', animation: 'spin 0.7s linear infinite' },
}

const resultStyles = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: '16px' },
  banner: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', padding: '16px 20px', borderRadius: '8px', border: '1px solid' },
  bannerLeft: { display: 'flex', alignItems: 'center', gap: '12px' },
  bannerLabel: { fontSize: '11px', color: 'var(--text-3)', fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '2px' },
  bannerPrediction: { fontFamily: 'var(--display)', fontSize: '22px', fontWeight: 700, lineHeight: 1.1 },
  bannerRight: { textAlign: 'right' },
  confidence: { fontFamily: 'var(--mono)', fontSize: '26px', fontWeight: 500, lineHeight: 1 },
  severityTag: { display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '5px', fontSize: '11px', color: 'var(--text-2)', fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: '4px' },
  severityDot: { width: '6px', height: '6px', borderRadius: '50%', flexShrink: 0 },
  meta: { display: 'flex', alignItems: 'center', gap: '8px', padding: '0 2px' },
  metaText: { fontSize: '12px', color: 'var(--text-3)' },
  metaMono: { fontFamily: 'var(--mono)', color: 'var(--text-2)' },
  metaDivider: { width: '1px', height: '12px', background: 'var(--border)' },
  scoresSection: { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px', padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: '12px' },
  sectionLabel: { fontSize: '11px', fontFamily: 'var(--mono)', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em' },
  scoresList: { display: 'flex', flexDirection: 'column', gap: '8px' },
  scoreRow: { display: 'grid', gridTemplateColumns: '140px 1fr 52px', alignItems: 'center', gap: '10px' },
  scoreLabel: { display: 'flex', alignItems: 'center', gap: '4px', fontSize: '13px', overflow: 'hidden', whiteSpace: 'nowrap' },
  barTrack: { height: '6px', background: 'var(--surface-2)', borderRadius: '3px', overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: '3px', minWidth: '2px' },
  scorePct: { fontFamily: 'var(--mono)', fontSize: '12px', textAlign: 'right' },
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
