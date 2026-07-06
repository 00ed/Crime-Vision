import { useState, useCallback } from 'react'
import axios from 'axios'

export function usePredict() {
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

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
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'Prediction failed. Is the backend running?'
      setError(msg)
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
