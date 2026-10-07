// Teacher + Student dashboards.
// Dev:   cd dashboard && npm install && npm run dev   (needs `python api/app.py` running)
// Build: npm run build  -> Flask serves dashboard/dist at http://localhost:5000
//
// Owner: Mayank Rathore (dashboard UI)

import { useEffect, useRef, useState } from "react";
import { get, post } from "./api.js";

const COLORS = { Distinction: "var(--dist)", Pass: "var(--pass)", Fail: "var(--risk)", Withdrawn: "var(--warn)" };
const level = (r) => (r >= 0.6 ? "High" : r >= 0.4 ? "Medium" : "Low");



function Bars({ items, wide }) {
  return (
    <div className={"bars" + (wide ? " wide" : "")}>
      {items.map(([label, frac, color, text]) => (
        <div className="row" key={label}>
          <span>{label}</span>
          <div className="track"><div className="fill" style={{ width: `${(frac * 100).toFixed(1)}%`, background: color }} /></div>
          <span>{text}</span>
        </div>
      ))}
    </div>
  );
}

function Kpis({ s }) {
  const sel = s.gbm.models[s.gbm.selected];
  const items = [
    [s.rf.rmse.toFixed(1), "RMSE, assessment grade", `Random Forest · R² ${s.rf.r2.toFixed(2)}`],
    [s.rf.mae.toFixed(1), "Average error in marks", "out of 100"],
    [(sel.accuracy * 100).toFixed(0) + "%", "Outcome accuracy", `${s.gbm.selected} · 4 classes`],
    [sel.f1_macro.toFixed(2), "Macro F1", `${s.gbm.n_test.toLocaleString()} held-out students`],
  ];
  return (
    <section className="grid kpis">
      {items.map((i) => (
        <div className="card kpi" key={i[1]}><b>{i[0]}</b><span>{i[1]}</span><small>{i[2]}</small></div>
      ))}
    </section>
  );
}

function Alerts({ rows }) {
  return (
    <section className="card stack">
      <h2>Early-warning alerts</h2>
      <p className="note">Held-out students whose predicted chance of failing or withdrawing is 40% or more, using activity up to day 100. The last column is what actually happened.</p>
      <div className="scroll">
        <table>
          <thead><tr><th>Student</th><th>Module</th><th>Clicks</th><th>Active days</th><th>Avg score</th><th>Risk</th><th>Actual result</th></tr></thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.id_student + s.code_module}>
                <td>{s.id_student}</td><td>{s.code_module}</td><td>{s.total_clicks}</td><td>{s.active_days}</td>
                <td>{Math.round(s.avg_score)}</td>
                <td><span className={"tag " + level(s.risk)}>{level(s.risk)} · {(s.risk * 100).toFixed(0)}%</span></td>
                <td>{s.final_result}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Comparison({ s }) {
  const rows = [["Linear Regression", s.baselines["Linear Regression"]], ["Decision Tree", s.baselines["Decision Tree"]], ["Random Forest", s.rf]];
  const best = Math.min(...rows.map((r) => r[1].rmse));
  const [imp, setImp] = useState("rf");
  const list = imp === "rf" ? s.rf.feature_importance : s.gbm.models[s.gbm.selected].feature_importance;
  return (
    <section className="grid two stack">
      <div className="card">
        <h2>Model comparison</h2>
        <p className="note">Assessment-grade regression (lower RMSE is better), students the models never saw.</p>
        <table className="cmp">
          <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead>
          <tbody>{rows.map(([n, m]) => (
            <tr key={n}><td>{n}{m.rmse === best ? " ✓" : ""}</td><td>{m.rmse.toFixed(2)}</td><td>{m.mae.toFixed(2)}</td><td>{m.r2.toFixed(3)}</td></tr>
          ))}</tbody>
        </table>
        <h3 style={{ marginTop: 20, fontSize: 16 }}>Final-outcome classification</h3>
        <table className="cmp">
          <thead><tr><th>Model</th><th>Accuracy</th><th>Macro F1</th></tr></thead>
          <tbody>{Object.entries(s.gbm.models).map(([n, m]) => (
            <tr key={n}><td>{n}{n === s.gbm.selected ? " ✓" : ""}</td><td>{(m.accuracy * 100).toFixed(1)}%</td><td>{m.f1_macro.toFixed(3)}</td></tr>
          ))}</tbody>
        </table>
      </div>
      <div className="card">
        <h2>What drives the predictions</h2>
        <div className="seg" role="group" aria-label="Choose model">
          <button aria-pressed={imp === "rf"} onClick={() => setImp("rf")}>Random Forest (grade)</button>
          <button aria-pressed={imp === "gbm"} onClick={() => setImp("gbm")}>Gradient Boosting (outcome)</button>
        </div>
        <Bars wide items={list.slice(0, 10).map((i) => [i.feature, i.importance / list[0].importance, "var(--dist)", (i.importance * 100).toFixed(0) + "%"])} />
      </div>
    </section>
  );
}

// ---- Student view: two live what-if predictors ----
function Controls({ spec, cats, choices, state, setState }) {
  return (
    <>
      {spec.map(([k, label, mn, mx, st]) => (
        <div key={k}>
          <label htmlFor={k}>{label}<output>{Math.round(state[k])}</output></label>
          <input id={k} type="range" min={mn} max={mx} step={st} value={Math.min(state[k], mx)}
                 onChange={(e) => setState({ ...state, [k]: +e.target.value })} />
        </div>
      ))}
      {cats.map(([k, label]) => (
        <div key={k}>
          <label htmlFor={k}>{label}</label>
          <select id={k} value={state[k]} onChange={(e) => setState({ ...state, [k]: e.target.value })}>
            {choices[k].map((c) => <option key={c}>{c}</option>)}
          </select>
        </div>
      ))}
    </>
  );
}

function useDebouncedPost(path, features, onResult) {
  const t = useRef();
  useEffect(() => {
    clearTimeout(t.current);
    t.current = setTimeout(() => post(path, { features }).then(onResult).catch(() => onResult(null)), 120);
    return () => clearTimeout(t.current);
  }, [JSON.stringify(features)]); // eslint-disable-line
}

function OutcomePredictor({ form }) {
  const [state, setState] = useState(form.defaults);
  const [res, setRes] = useState(null);
  useDebouncedPost("/predict/outcome", state, setRes);
  const spec = [["total_clicks", "VLE clicks so far", 0, 3000, 10], ["active_days", "Active days on the VLE", 0, 100, 1],
                ["avg_score", "Average assessment score", 0, 100, 1], ["n_submitted", "Assessments submitted", 0, 8, 1],
                ["clicks_last_30d", "Clicks in the last 30 days", 0, 1500, 10]];
  const cats = [["code_module", "Module"], ["highest_education", "Highest education"], ["age_band", "Age band"]];
  return (
    <section className="grid two">
      <div className="card">
        <h2>Will this learner finish?</h2>
        <p className="note">Change the learner's activity up to course day 100 and watch the predicted outcome move.</p>
        <Controls spec={spec} cats={cats} choices={form.choices} state={state} setState={setState} />
      </div>
      <div className="card">
        <h2>Predicted final outcome</h2>
        {res ? (
          <>
            <Bars items={Object.entries(res.probabilities).map(([c, p]) => [c, p, COLORS[c], (p * 100).toFixed(0) + "%"])} />
            <div className={"verdict " + res.risk_level}>{res.risk_level} risk · {(res.risk * 100).toFixed(0)}% chance of failing or withdrawing</div>
            <p className="note">Most likely: {res.predicted_outcome} ({(res.confidence * 100).toFixed(0)}%). Training used balanced class weights so rare outcomes are not ignored.</p>
          </>
        ) : <p className="note">Waiting for the API…</p>}
      </div>
    </section>
  );
}

function GradePredictor({ form }) {
  const [state, setState] = useState(form.defaults);
  const [res, setRes] = useState(null);
  useDebouncedPost("/predict/grade", state, setRes);
  const spec = [["prev_score", "Score on previous assessment", 0, 100, 1], ["prev_avg_score", "Average of earlier scores", 0, 100, 1],
                ["n_prev_assessments", "Assessments already done", 0, 8, 1], ["cum_clicks", "VLE clicks so far", 0, 4000, 10],
                ["cum_active_days", "Active days so far", 0, 150, 1], ["weight", "Weight of this assessment (%)", 0, 100, 5]];
  const cats = [["assessment_type", "Assessment type"], ["code_module", "Module"]];
  return (
    <section className="grid two stack">
      <div className="card">
        <h2>What grade is next?</h2>
        <p className="note">Random Forest regressor: forecasts the score on the learner's next assessment.</p>
        <Controls spec={spec} cats={cats} choices={form.choices} state={state} setState={setState} />
      </div>
      <div className="card">
        <h2>Predicted grade</h2>
        {res ? (<><div className="big">{res.predicted_score.toFixed(0)}<small style={{ fontSize: 20, color: "var(--muted)" }}> / 100</small></div>
          <p className="note">Typical error on unseen students is about ±10 marks (MAE).</p></>) : <p className="note">Waiting for the API…</p>}
      </div>
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState("teacher");
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    Promise.all([get("/dashboard/summary"), get("/dashboard/alerts")])
      .then(([s, a]) => { setSummary(s); setAlerts(a.alerts); })
      .catch(() => setError(true));
  }, []);

  return (
    <div className="wrap">
      <header>
        <div>
          <h1>Student Performance Prediction in Online Courses</h1>
          <p>Predict assessment grades and final learner outcomes using machine-learning models trained on the Open University Learning Analytics Dataset (OULAD).</p>
        </div>
      </header>

      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "teacher"} onClick={() => setTab("teacher")}>Teacher view</button>
        <button role="tab" aria-selected={tab === "student"} onClick={() => setTab("student")}>Student view</button>
      </div>

      {error && <div className="error">Could not reach the prediction API. Is <code>python api/app.py</code> running?</div>}
      {summary && tab === "teacher" && (<><Kpis s={summary} /><Alerts rows={alerts} /><Comparison s={summary} /></>)}
      {summary && tab === "student" && (<><OutcomePredictor form={summary.forms.outcome} /><GradePredictor form={summary.forms.grade} /></>)}
    </div>
  );
}
