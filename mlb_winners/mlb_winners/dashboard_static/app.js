const $ = (s) => document.querySelector(s),
  pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
let busy = false,
  detailGame = null;
function diamond(mask) {
  return `<div class="diamond"><i class="base b1 ${mask & 1 ? "on" : ""}"></i><i class="base b2 ${mask & 2 ? "on" : ""}"></i><i class="base b3 ${mask & 4 ? "on" : ""}"></i></div>`;
}
function pitchDistribution(items) {
  return (items || [])
    .slice(0, 3)
    .map((x) => `${x.pitch_type} ${(x.probability * 100).toFixed(0)}%`)
    .join(" · ");
}
function card(g) {
  const m = g.market,
    edge = g.home_edge >= g.away_edge ? g.home_edge : g.away_edge;
  return `<article class="game-card ${g.actionable_side ? "actionable" : ""}" data-game="${g.game_pk}">
 <div class="game-head"><span class="inning">${g.half_inning} ${g.inning} · ${g.outs} out</span><span class="count">${g.balls}–${g.strikes}</span></div>
 <div class="scoreboard"><span class="team">${g.away_team}</span><span class="score">${g.away_score}</span><span class="team">${g.home_team}</span><span class="score">${g.home_score}</span></div>
 <div class="matchup"><span><strong>${g.batter_name || "Unknown batter"}</strong><br>vs ${g.pitcher_name || "Unknown pitcher"}</span>${diamond(g.base_mask)}</div>
 <div class="probabilities"><div class="prob-row"><span>${g.home_team}</span><b>${pct(g.home_win_prob)}</b></div><div class="bar"><i style="width:${g.home_win_prob * 100}%"></i></div>
 <div class="prob-row"><span>${g.away_team}</span><b>${pct(g.away_win_prob)}</b></div><div class="bar away"><i style="width:${g.away_win_prob * 100}%"></i></div></div>
 <div class="market-row"><span>Score this inning <b>${pct(g.score_this_inning_prob)}</b></span><span>${m ? `${m.bookmaker} ${m.stale ? '<em class="stale">STALE</em>' : `· edge <b class="${edge >= 0.05 ? "edge" : ""}">${pct(edge)}</b>`}` : "No live odds"}</span></div>
 ${g.most_likely_pitch ? `<div class="next-pitch"><span>Highest-probability pitch <b>${g.most_likely_pitch}</b> · ${pct(g.most_likely_pitch_prob)}</span><small>Probability distribution · ${pitchDistribution(g.top_three_pitches)}</small></div>` : ""}
 ${(g.quality_flags || []).length ? `<div class="flag">${(g.quality_flags || []).join(" · ")}</div>` : ""}</article>`;
}
async function loadSlate() {
  try {
    const r = await fetch("/api/live/games"),
      d = await r.json();
    $("#feedStatus").textContent = "Live · 5 sec";
    $("#gameCount").textContent = d.games.length;
    $("#updatedAt").textContent =
      `Updated ${new Date(d.generated_at).toLocaleTimeString()}`;
    $("#games").innerHTML = d.games.length
      ? d.games.map(card).join("")
      : '<div class="empty">No live games right now. The dashboard will pick them up automatically.</div>';
    document
      .querySelectorAll(".game-card")
      .forEach((el) => (el.onclick = () => showGame(el.dataset.game)));
  } catch (e) {
    showError(e.message);
  }
}
async function loadOddsStatus() {
  const d = await fetch("/api/live/odds/status").then((r) => r.json());
  const b = $("#refreshOdds");
  b.disabled = busy || d.cooldown_seconds > 0;
  b.textContent =
    d.cooldown_seconds > 0
      ? `Refresh in ${d.cooldown_seconds}s`
      : "Refresh live odds";
  $("#oddsStatus").textContent =
    d.status === "never"
      ? "Odds never refreshed"
      : `${d.status} · ${d.rows || 0} rows · remaining ${d.remaining ?? "?"}`;
}
async function refreshOdds() {
  if (
    !confirm(
      "This makes one Odds API request and consumes monthly quota. Refresh the entire MLB slate now?",
    )
  )
    return;
  busy = true;
  await loadOddsStatus();
  try {
    const r = await fetch("/api/live/odds/refresh", {
        method: "POST",
        headers: { "X-Manual-Odds-Refresh": "confirmed" },
      }),
      d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Odds refresh failed");
    await loadSlate();
  } catch (e) {
    showError(e.message);
  } finally {
    busy = false;
    await loadOddsStatus();
  }
}
async function showGame(id) {
  detailGame = id;
  history.pushState({ game: id }, "", `/game/${id}`);
  $("#slateView").classList.add("hidden");
  $("#detailView").classList.remove("hidden");
  await loadDetail();
}
async function loadDetail() {
  if (!detailGame) return;
  try {
    const r = await fetch(`/api/live/games/${detailGame}`),
      d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    const g = d.game,
      p = d.pitch_analysis || {},
      pitchRows = (p.pitches || []).filter((row) => row.is_plausible !== false),
      sens = d.pitch_sensitivities || [];
    $("#detail").innerHTML =
      `<div class="detail-hero"><div class="detail-panel"><p class="eyebrow">${g.half_inning} ${g.inning} · ${g.outs} OUT · COUNT ${g.balls}–${g.strikes}</p><h2>${g.away_team} ${g.away_score} — ${g.home_team} ${g.home_score}</h2><div class="big-prob">${pct(g.home_win_prob)}</div><p>${g.home_team} win probability · score this inning ${pct(g.score_this_inning_prob)}</p></div><div class="detail-panel"><p class="eyebrow">CURRENT MATCHUP</p><h2>${g.batter_name || "Unknown"} vs ${g.pitcher_name || "Unknown"}</h2><div style="display:flex;align-items:center;margin-top:24px">${diamond(g.base_mask)}<p>State ${g.outs * 8 + g.base_mask + 1} of 24 · ${g.pitcher_pitch_count || 0} pitches<br><span class="stale">${(g.quality_flags || []).join(" · ")}</span></p></div></div></div>${
        pitchRows.length
          ? `<div class="detail-panel timeline pitch-lab"><div class="pitch-title"><div><p class="eyebrow">PITCH-TYPE LIVE SIMULATION</p><h2>Highest-probability pitch: ${p.most_likely_pitch_name || p.most_likely_pitch} · ${pct(p.most_likely_pitch_prob)}</h2></div><span>${g.batter_side || "?"} batter · ${g.pitcher_hand || "?"} pitcher</span></div><p class="distribution-note">Probability distribution · ${pitchDistribution(p.top_three)}. Simulations sample all pitches, not only the leader.</p><div class="pitch-table"><div class="pitch-table-head"><span>Pitch</span><span>Next</span><span>Expected / actual</span><span>Velocity</span><span>Batter result</span><span>Win sensitivity</span></div>${pitchRows
              .map((row) => {
                const s = sens.find((x) => x.pitch_type === row.pitch_type);
                return `<div class="pitch-table-row"><b>${row.pitch_name}</b><strong>${pct(row.probability)}</strong><span>${pct(row.expected_usage)} / ${pct(row.actual_usage)}</span><span>${row.current_velocity ? Number(row.current_velocity).toFixed(1) : row.expected_velocity ? Number(row.expected_velocity).toFixed(1) : "—"} mph ${row.velocity_delta != null ? `<small class="${row.velocity_delta < 0 ? "down" : "up"}">${row.velocity_delta >= 0 ? "+" : ""}${Number(row.velocity_delta).toFixed(1)}</small>` : ""}</span><span>${row.expected_woba != null ? Number(row.expected_woba).toFixed(3) + " xwOBA" : pct(row.whiff_rate) + " whiff"}</span><span>${s ? pct(s.home_win_prob) : "—"} ${s ? `<small class="${s.delta < 0 ? "down" : "up"}">${s.delta >= 0 ? "+" : ""}${(s.delta * 100).toFixed(1)}pp</small>` : ""}</span></div>`;
              })
              .join("")}</div></div>`
          : ""
      }<div class="detail-panel timeline"><p class="eyebrow">24-STATE SCORING MATRIX · ${d.state_rates.games_used}/50 GAMES</p><div class="matrix">${d.state_rates.states.map((s) => `<div class="cell ${s.base_mask === g.base_mask && s.outs === g.outs ? "current" : ""}"><small>${s.outs} out · ${s.base_mask}</small><b>${pct(s.score_probability)}</b><small>n=${s.sample_size}</small></div>`).join("")}</div></div><div class="detail-panel timeline"><p class="eyebrow">RECENT PITCH EVENTS</p>${d.events.map((e) => `<div class="event"><time>${e.half_inning} ${e.inning} · ${e.balls}-${e.strikes}</time><span>${e.event_description || e.pitch_type || "Game event"} ${e.start_speed ? `· ${Number(e.start_speed).toFixed(1)} mph` : ""}</span></div>`).join("") || "<p>No events stored yet.</p>"}</div>`;
  } catch (e) {
    showError(e.message);
  }
}
function showError(msg) {
  $("#error").textContent = msg;
  $("#error").classList.remove("hidden");
  setTimeout(() => $("#error").classList.add("hidden"), 6000);
}
$("#refreshOdds").onclick = refreshOdds;
$("#backButton").onclick = () => {
  detailGame = null;
  history.pushState({}, "", "/");
  $("#detailView").classList.add("hidden");
  $("#slateView").classList.remove("hidden");
};
window.onpopstate = () => {
  const m = location.pathname.match(/game\/(\d+)/);
  detailGame = m?.[1] || null;
  if (detailGame) {
    $("#slateView").classList.add("hidden");
    $("#detailView").classList.remove("hidden");
    loadDetail();
  } else {
    $("#detailView").classList.add("hidden");
    $("#slateView").classList.remove("hidden");
  }
};
const initial = location.pathname.match(/game\/(\d+)/);
if (initial) {
  detailGame = initial[1];
  $("#slateView").classList.add("hidden");
  $("#detailView").classList.remove("hidden");
  loadDetail();
}
loadSlate();
loadOddsStatus();
setInterval(() => {
  detailGame ? loadDetail() : loadSlate();
  loadOddsStatus();
}, 5000);
