import { PlayEditor } from '@/frontend/editor/components/PlayEditor'

export default function HomePage() {
  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">NBA</span>
          <span>
            <strong>Play Lab</strong>
            <small>Structured basketball simulation</small>
          </span>
        </div>
        <span className="phase-label">Interactive play editor</span>
      </header>

      <section className="workspace-intro">
        <p className="eyebrow">Basketball whiteboard</p>
        <h1>Draw the play.<br />Watch it react.</h1>
        <p>
          Drag players to sketch basketball actions. The editor understands
          cuts, dribbles, passes, shots, screens, and automatic defense.
        </p>
      </section>

      <PlayEditor />
    </main>
  )
}
