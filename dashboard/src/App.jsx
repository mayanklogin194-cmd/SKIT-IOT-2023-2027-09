// Teacher + Student dashboard skeleton (Sprint 7).
// Scaffold with: npx create-vite@latest dashboard -- --template react
// then drop this in as src/App.jsx.
//
// Owner: Priyanshu Joshi (Sprint 7 — dashboard UI)

import { useEffect, useState } from "react";

const API_BASE = "http://localhost:5000";

function StudentCard({ student }) {
  return (
    <div className="student-card">
      <h3>{student.name}</h3>
      <p>Predicted grade: {student.predictedScore ?? "—"}</p>
      <p>Predicted outcome: {student.predictedOutcome ?? "—"}</p>
      {student.atRisk && <span className="risk-badge">Early warning</span>}
    </div>
  );
}

export default function App() {
  const [alerts, setAlerts] = useState([]);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    fetch(`${API_BASE}/dashboard/alerts`)
      .then((res) => res.json())
      .then((data) => {
        setAlerts(data.alerts || []);
        setStatus("ready");
      })
      .catch(() => setStatus("error"));
  }, []);

  return (
    <div className="app">
      <header>
        <h1>Student Performance Dashboard</h1>
      </header>

      {status === "loading" && <p>Loading…</p>}
      {status === "error" && <p>Could not reach the prediction API. Is api/app.py running?</p>}

      <section className="alerts">
        <h2>Early-warning alerts (teacher view)</h2>
        {alerts.length === 0 ? (
          <p>No at-risk students flagged yet — connect this to real predictions.</p>
        ) : (
          alerts.map((s) => <StudentCard key={s.id} student={s} />)
        )}
      </section>
    </div>
  );
}
