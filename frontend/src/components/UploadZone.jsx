import { useRef, useState, useCallback } from 'react'
import { Upload, Film, X } from 'lucide-react'

const ACCEPTED = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska']

export default function UploadZone({ onFile, loading, currentFile, onClear }) {
  const inputRef   = useRef(null)
  const [drag, setDrag] = useState(false)
  const videoRef   = useRef(null)

  const handleFile = useCallback((file) => {
    if (!file) return
    if (!ACCEPTED.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
      alert('Unsupported format. Please upload mp4, avi, mov, or mkv.')
      return
    }
    if (file.size > 100 * 1024 * 1024) {
      alert('File too large. Max 100 MB.')
      return
    }
    onFile(file)
  }, [onFile])

  const onDrop = (e) => {
    e.preventDefault()
    setDrag(false)
    handleFile(e.dataTransfer.files[0])
  }

  const previewUrl = currentFile ? URL.createObjectURL(currentFile) : null

  return (
    <div style={styles.wrapper}>
      {/* Drop zone */}
      <div
        style={{
          ...styles.zone,
          ...(drag ? styles.zoneDrag : {}),
          ...(loading ? styles.zoneLoading : {}),
        }}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        onClick={() => !currentFile && !loading && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {/* Scan line when loading */}
        {loading && <div style={styles.scanLine} />}

        {currentFile ? (
          <div style={styles.preview}>
            <video
              ref={videoRef}
              src={previewUrl}
              style={styles.video}
              muted
              loop
              autoPlay
              playsInline
            />
            <div style={styles.previewOverlay}>
              <Film size={14} color="var(--amber)" />
              <span style={styles.filename}>{currentFile.name}</span>
              <span style={styles.filesize}>
                {(currentFile.size / (1024 * 1024)).toFixed(1)} MB
              </span>
            </div>
            {!loading && (
              <button
                style={styles.clearBtn}
                onClick={(e) => { e.stopPropagation(); onClear() }}
                title="Remove video"
              >
                <X size={14} />
              </button>
            )}
          </div>
        ) : (
          <div style={styles.emptyState}>
            <div style={styles.iconRing}>
              <Upload size={22} color="var(--amber)" strokeWidth={1.5} />
            </div>
            <p style={styles.headline}>Drop a surveillance clip here</p>
            <p style={styles.sub}>or click to browse — mp4, avi, mov, mkv · max 100 MB</p>
          </div>
        )}
      </div>

      {/* Analyse button */}
      {currentFile && !loading && (
        <button
          style={styles.analyseBtn}
          onClick={() => onFile(currentFile)}
          className="fade-up"
        >
          <span style={styles.btnDot} />
          Analyse clip
        </button>
      )}

      {loading && (
        <div style={styles.analysing} className="fade-up">
          <span style={styles.spinner} />
          Analysing footage…
        </div>
      )}
    </div>
  )
}

const styles = {
  wrapper: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
  },
  zone: {
    position: 'relative',
    border: '1.5px dashed var(--border-hover)',
    borderRadius: '12px',
    background: 'var(--surface)',
    minHeight: '260px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    overflow: 'hidden',
    transition: 'border-color 0.2s, background 0.2s',
  },
  zoneDrag: {
    borderColor: 'var(--amber)',
    background: 'var(--amber-glow-s)',
  },
  zoneLoading: {
    cursor: 'default',
    borderColor: 'var(--amber-dim)',
  },
  scanLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: '2px',
    background: 'linear-gradient(90deg, transparent, var(--amber), transparent)',
    animation: 'scan 1.8s linear infinite',
    zIndex: 10,
    pointerEvents: 'none',
  },
  preview: {
    width: '100%',
    height: '100%',
    position: 'relative',
  },
  video: {
    width: '100%',
    height: '240px',
    objectFit: 'cover',
    display: 'block',
    opacity: 0.7,
  },
  previewOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: '8px 12px',
    background: 'linear-gradient(transparent, rgba(0,0,0,0.8))',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontFamily: 'var(--mono)',
    fontSize: '11px',
    color: 'var(--text-2)',
  },
  filename: {
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    color: 'var(--text-1)',
  },
  filesize: {
    color: 'var(--text-3)',
    flexShrink: 0,
  },
  clearBtn: {
    position: 'absolute',
    top: '10px',
    right: '10px',
    background: 'rgba(0,0,0,0.6)',
    border: '1px solid var(--border-hover)',
    borderRadius: '6px',
    color: 'var(--text-2)',
    cursor: 'pointer',
    width: '28px',
    height: '28px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'color 0.15s, border-color 0.15s',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
    padding: '40px 20px',
    userSelect: 'none',
  },
  iconRing: {
    width: '56px',
    height: '56px',
    borderRadius: '50%',
    border: '1.5px solid var(--border-hover)',
    background: 'var(--amber-glow-s)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headline: {
    fontFamily: 'var(--display)',
    fontSize: '16px',
    fontWeight: 600,
    color: 'var(--text-1)',
  },
  sub: {
    fontSize: '13px',
    color: 'var(--text-3)',
    textAlign: 'center',
    maxWidth: '300px',
  },
  analyseBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10px',
    width: '100%',
    padding: '13px',
    background: 'var(--amber)',
    color: '#0B0E0E',
    fontFamily: 'var(--display)',
    fontWeight: 700,
    fontSize: '15px',
    border: 'none',
    borderRadius: '10px',
    cursor: 'pointer',
    letterSpacing: '0.02em',
    transition: 'opacity 0.15s',
  },
  btnDot: {
    width: '7px',
    height: '7px',
    borderRadius: '50%',
    background: '#0B0E0E',
    opacity: 0.6,
  },
  analysing: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10px',
    color: 'var(--amber)',
    fontFamily: 'var(--mono)',
    fontSize: '13px',
    padding: '10px',
  },
  spinner: {
    display: 'inline-block',
    width: '14px',
    height: '14px',
    border: '2px solid var(--amber-glow)',
    borderTopColor: 'var(--amber)',
    borderRadius: '50%',
    animation: 'spin 0.7s linear infinite',
  },
}
