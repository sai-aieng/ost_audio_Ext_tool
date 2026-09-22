import { useEffect, useRef, useState } from 'react';
import { DropZone } from './DropZone';
import { analyzeWithGemini, getGeminiConfig, getGeminiStatus, getGeminiResults, geminiDownloadUrl, geminiFaceUrl } from '../api/client';

function time(seconds) {
  const total = Math.floor(seconds);
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

export function GeminiAnalysisPanel() {
  const [configuration, setConfiguration] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [options, setOptions] = useState({ dialogue: true, faces: true });
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [tab, setTab] = useState('scenes');
  const [retry, setRetry] = useState(0);
  const video = useRef(null);
  const busy = submitting || Boolean(job && !['completed', 'failed'].includes(status?.status));

  useEffect(() => {
    let active = true;
    getGeminiConfig().then(value => { if (active) setConfiguration(value); })
      .catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [retry]);

  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!job) return;
    let stopped = false;
    let timer;
    let failures = 0;
    async function poll() {
      try {
        const next = await getGeminiStatus(job);
        if (stopped) return;
        setStatus(next);
        if (next.status === 'completed') {
          const data = await getGeminiResults(job);
          if (!stopped) { setResult(data); setError(''); }
          return;
        }
        if (next.status === 'failed') { setError(next.error || 'Analysis failed.'); return; }
        failures = 0;
        setError('');
      } catch (e) {
        if (stopped) return;
        failures += 1;
        setError(e.message);
        if (failures >= 3) return;
      }
      if (!stopped) timer = setTimeout(poll, 2500);
    }
    void poll();
    return () => { stopped = true; clearTimeout(timer); };
  }, [job, retry]);

  function selectFile(next) {
    if (!/\.(mp4|mov|mkv|avi)$/i.test(next.name)) { setError('Choose an MP4, MOV, MKV, or AVI video.'); return; }
    if (next.size > 500 * 1024 * 1024) { setError('Choose a video smaller than 500 MB.'); return; }
    setFile(next); setJob(null); setStatus(null); setResult(null); setError('');
  }

  async function analyze() {
    setSubmitting(true); setError(''); setResult(null); setStatus(null); setJob(null); setTab('scenes');
    try {
      const started = await analyzeWithGemini(file, options);
      setStatus({ status: 'queued', stage: 'Video uploaded. Starting analysis...', elapsed_sec: 0 });
      setJob(started.job_id);
    } catch (e) { setError(e.message); }
    finally { setSubmitting(false); }
  }

  function seek(seconds) {
    if (video.current) { video.current.currentTime = seconds; video.current.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
  }

  return <div className="gemini-workspace">
    <section className="gemini-hero">
      <div><span className="gemini-kicker">GOOGLE GEMINI / VIDEO ANALYSIS</span>
        <h2>Your video, scene by scene.</h2>
        <p>Connect the dialogue, visuals and on-screen text in one timestamped report.</p></div>
      <span className={`gemini-connection ${configuration?.configured ? 'ready' : ''}`}>
        {configuration ? (configuration.configured ? 'API key configured' : 'API key required') : 'Checking connection settings...'}
      </span>
    </section>
    <div className="gemini-setup">
      <section className="card gemini-upload">
        <div className="section-heading"><div><p className="eyebrow">SOURCE VIDEO</p><h3>Analyze a video</h3></div><span className="gemini-tag">Up to 20 min</span></div>
        <DropZone selectedFile={file} onFileSelect={selectFile} disabled={busy} label="Choose video for Gemini analysis" />
        <p className="gemini-note">MP4, MOV, MKV or AVI · up to 500 MB. The video is sent to Google when you start analysis.</p>
        <fieldset className="gemini-options" disabled={busy}>
          <legend>Include in your report</legend>
          <label><input type="checkbox" checked={options.dialogue} onChange={e => setOptions({ ...options, dialogue: e.target.checked })} /><span><strong>Transcribed dialogue</strong><small>Use the existing Whisper engine for speech and timing.</small></span></label>
          <label><input type="checkbox" checked={options.faces} onChange={e => setOptions({ ...options, faces: e.target.checked })} /><span><strong>Face photos</strong><small>Crop faces from scene samples and filter duplicate photos.</small></span></label>
        </fieldset>
        <button type="button" className="primary-button gemini-start" disabled={!file || busy || !configuration?.configured} onClick={analyze}>
          {busy ? <><span className="button-spinner" aria-hidden="true" />{submitting ? 'Uploading video...' : 'Analysis in progress...'}</> : 'Analyze with Gemini'}
        </button>
        {error && <div className="error-message" role="alert">{error}<button type="button" className="secondary-button" onClick={() => { setError(''); setRetry(x => x + 1); }}>Retry connection / status</button></div>}
        {configuration && !configuration.configured && <p role="alert">Add GOOGLE_API_KEY to backend/.env, or to the Render service environment.</p>}
      </section>
      <section className="card gemini-preview">
        {preview ? <video ref={video} src={preview} controls preload="metadata" /> : <div className="gemini-preview-empty"><span aria-hidden="true">▷</span><h3>Preview your source</h3><p>Select a video to preview it here.<br />Click a result timestamp to revisit a scene.</p></div>}
        <div className="gemini-preview-caption"><strong>{file?.name || 'No video selected'}</strong><small>{configuration?.model || 'Gemini visual analysis'} · Whisper dialogue</small></div>
        <div className="gemini-report-guide"><strong>One report. Five connected columns.</strong><p>Time range · Dialogue · Visual description · Purpose / context · On-screen text</p></div>
      </section>
    </div>
    {status && <section className="gemini-progress" aria-live="polite"><div><strong>{status.stage}</strong><p>{status.status === 'completed' ? 'Your report is ready to review and download.' : 'You can switch tools while this analysis runs.'}</p></div><span>{Math.round(status.elapsed_sec || 0)}s elapsed</span></section>}
    <section className="card gemini-results">
      <div className="section-heading"><div><p className="eyebrow">ANALYSIS RESULTS</p><h3>{result ? `${result.scenes.length} scenes across ${time(result.duration_sec)}` : 'Your video report'}</h3></div>
        {result && <div className="download-actions"><a className="secondary-button" href={geminiDownloadUrl(job, 'csv')}>Download CSV</a><a className="secondary-button" href={geminiDownloadUrl(job, 'json')}>Download JSON</a></div>}
      </div>
      <div className="gemini-result-tabs" role="group" aria-label="Analysis results">
        <button type="button" aria-pressed={tab === 'scenes'} onClick={() => setTab('scenes')}>Scene table {result && <span>{result.scenes.length}</span>}</button>
        <button type="button" aria-pressed={tab === 'faces'} onClick={() => setTab('faces')}>Face photos {result && <span>{result.faces.length}</span>}</button>
      </div>
      {!result ? <div className="gemini-empty"><h3>{busy ? 'Building your report...' : 'A clear view of every scene'}</h3><p>{busy ? 'Progress updates appear above. Results will appear automatically.' : 'Upload a video and start analysis to see the timestamped table and face photos.'}</p></div> : <>
        {tab === 'scenes' ? <div className="gemini-table-scroll" tabIndex="0" aria-label="Scrollable scene analysis table"><table className="gemini-table"><thead><tr>{['Timestamp Range', 'Transcribed Dialogue', 'Visual Scene Description', 'Purpose / Context of Visual', 'On-Screen Text'].map(text => <th key={text} scope="col">{text}</th>)}</tr></thead><tbody>
          {result.scenes.map((row, index) => <tr key={index}><td><button type="button" className="gemini-time" onClick={() => seek(row.start_sec)}>{time(row.start_sec)} – {time(row.end_sec)}</button></td><td>{row.transcribed_dialogue || (result.options.include_dialogue ? 'No dialogue' : 'Not requested')}</td><td>{row.visual_description}</td><td><span className="gemini-inferred">Interpretation</span>{row.purpose_context}</td><td>{row.on_screen_text.length ? row.on_screen_text.join('\n') : 'None'}</td></tr>)}
        </tbody></table></div> : <>
          <p className="gemini-note">Sampled face photos with observation times. These are not confirmed narrator identities or continuous appearance ranges.</p>
          {result.faces.length ? <div className="gemini-face-grid">{result.faces.map(face => <article key={face.id}><img src={geminiFaceUrl(job, face.id)} alt={`Extracted ${face.id}`} /><div><strong>{face.id.replace('face-', 'Face ')}</strong><p>{face.observed_at_sec.map((seconds, i) => <button type="button" className="gemini-time" key={i} onClick={() => seek(seconds)}>{time(seconds)}</button>)}</p><a href={geminiFaceUrl(job, face.id)} download>Download photo</a></div></article>)}</div> : <div className="gemini-empty"><h3>{result.options.include_faces ? 'No face photos returned' : 'Face extraction was not requested'}</h3><p>Review the analysis notes below for any processing issues.</p></div>}
        </>}
        <details className="gemini-notes"><summary>Analysis notes and limitations</summary><ul>{result.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul></details>
      </>}
    </section>
  </div>;
}
