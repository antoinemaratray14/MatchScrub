"""
Match Heatmap Explorer — StatsBomb events, scrubbable time windows.

Run it from a terminal (NOT from inside Spyder):

    streamlit run pressure_app.py

Metrics: pressures, ball receipts, OBV, and defensive actions. Drag the window
start to scrub through the match; the window length is a separate control so it
stays fixed while you scrub, and it cannot go below MIN_WINDOW minutes.

CREDENTIALS
-----------
Read from environment variables first, then .streamlit/secrets.toml, then the
sidebar. Don't commit them: add .streamlit/secrets.toml to .gitignore.

    export SB_USER="..."   export SB_PASS="..."

TWO THINGS THAT WOULD OTHERWISE MISLEAD
---------------------------------------
1. DENSITY IS PER MINUTE. Otherwise a 20-minute window looks five times more
   intense than a 2-minute one purely because it is longer, and scrubbing tells
   you nothing. With a per-minute rate and a fixed colour ceiling, windows of
   different lengths are directly comparable. "Scale to window" is available
   for looking at structure inside a quiet spell, but it makes windows
   non-comparable, which is why it is off by default.

2. SMALL SAMPLES ARE SHOWN AS POINTS. At the 2-minute minimum you are looking
   at a handful of events. Individual events are drawn on top and the count is
   displayed, so a smooth-looking blob built on four events is visibly built on
   four events.

DIRECTION: coordinates are normalised so the team performing the action attacks
toward x = 120. For pressures that means high x is pressing high up the pitch.
The two teams sit in opposite frames and cannot be overlaid, so pick one team.

OBV is summed, not counted, and can be negative — it uses a diverging colour
scale centred on zero. Red is value added, blue is value lost.
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
API_BASE = "https://data.statsbombservices.com/api"
EVENTS_VERSION = "v11"
MATCHES_VERSION = "v6"
COMPETITIONS_VERSION = "v4"

MIN_WINDOW = 2.0
DEFAULT_WINDOW = 15.0

BG = "#faf8f5"
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#faf8f5", "#f7d9c4", "#f0a07a", "#dc5b3c", "#a5281c", "#5e0f0a"])
DIVERGE = LinearSegmentedColormap.from_list(
    "div", ["#1f5f8b", "#8fbcd9", "#faf8f5", "#f0a07a", "#a5281c"])
PITCH_LINE = "#b0b0b0"

METRICS = {
    "Pressures": {
        "types": ["Pressure"], "value": None, "diverging": False,
        "help": "Pressure events. High x = pressing high up the pitch.",
    },
    "Ball receipts": {
        "types": ["Ball Receipt*"], "value": None, "diverging": False,
        "help": "Where the team receives the ball.",
    },
    "OBV (for, net)": {
        "types": None, "value": "obv_for_net", "diverging": True,
        "help": "Summed on-ball value. Red adds value, blue loses it.",
    },
    "Defensive actions": {
        "types": ["Pressure", "Duel", "Interception", "Block",
                  "Ball Recovery", "Foul Committed", "Clearance"],
        "value": None, "diverging": False,
        "help": "Pressures plus duels, interceptions, blocks, recoveries, "
                "fouls and clearances.",
    },
    "Passes": {
        "types": ["Pass"], "value": None, "diverging": False,
        "help": "Pass origins.",
    },
    "Carries": {
        "types": ["Carry"], "value": None, "diverging": False,
        "help": "Carry origins.",
    },
}

KEEP = ["id", "index", "period", "minute", "second", "type.name", "team.name",
        "player.name", "position.name", "location", "obv_for_net",
        "obv_against_net", "under_pressure", "counterpress"]


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def _get(url, user, pw, tries=4):
    for attempt in range(tries):
        r = requests.get(url, auth=HTTPBasicAuth(user, pw), timeout=60)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"failed after {tries} attempts: {url}")


@st.cache_data(show_spinner="Loading competitions…", ttl=3600)
def load_competitions(user, _pw):
    js = _get(f"{API_BASE}/{COMPETITIONS_VERSION}/competitions", user, _pw)
    return pd.DataFrame(js)


@st.cache_data(show_spinner="Loading fixtures…", ttl=3600)
def load_matches(comp_id, season_id, user, _pw):
    js = _get(f"{API_BASE}/{MATCHES_VERSION}/competitions/{comp_id}/"
              f"seasons/{season_id}/matches", user, _pw)
    return pd.json_normalize(js, sep=".")


@st.cache_data(show_spinner="Loading events…", ttl=3600, max_entries=20)
def load_events(match_id, user, _pw):
    js = _get(f"{API_BASE}/{EVENTS_VERSION}/events/{match_id}", user, _pw)
    df = pd.json_normalize(js, sep=".")
    return df.reindex(columns=[c for c in KEEP if c in df.columns])


@st.cache_data(show_spinner="Preparing events…", ttl=3600, max_entries=20)
def get_match_data(match_id, user, _pw):
    """
    Cached on match_id rather than on the events DataFrame. Caching on the
    frame itself makes Streamlit hash a column of [x, y] lists, which is
    unhashable — it falls back to pickling the whole frame on every widget
    change, which is slow and silent.
    """
    return prepare(load_events(match_id, user, _pw))


def prepare(df):
    """Elapsed-minute clock plus x/y. Returns (events, total_minutes, periods)."""
    d = df[pd.to_numeric(df["period"], errors="coerce") <= 4].copy()
    d["period"] = pd.to_numeric(d["period"], errors="coerce")
    d["minute"] = pd.to_numeric(d["minute"], errors="coerce")
    d["second"] = pd.to_numeric(d["second"], errors="coerce").fillna(0)
    d["clock"] = d["minute"] + d["second"] / 60.0
    if "obv_for_net" in d.columns:
        d["obv_for_net"] = pd.to_numeric(d["obv_for_net"], errors="coerce")

    # `minute` resets to 45 at half-time, so elapsed time must be built up
    # period by period or the timeline jumps backwards mid-match.
    g = d.groupby("period")["clock"].agg(["min", "max"])
    durs = (g["max"] - g["min"]).clip(lower=0)
    cum = durs.cumsum().shift(1).fillna(0.0)
    starts = g["min"]
    d["t"] = (d["period"].map(cum)
              + (d["clock"] - d["period"].map(starts)).clip(lower=0))

    xs = np.full(len(d), np.nan)
    ys = np.full(len(d), np.nan)
    for i, v in enumerate(d["location"].to_numpy()):
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                xs[i] = float(v[0]); ys[i] = float(v[1])
            except (TypeError, ValueError):
                pass
    d["x"], d["y"] = xs, ys
    return d, float(durs.sum()), durs.to_dict()


# ----------------------------------------------------------------------------
# PITCH + DENSITY
# ----------------------------------------------------------------------------
def draw_pitch(ax):
    ax.set_facecolor(BG)
    L = dict(color=PITCH_LINE, lw=1.0, zorder=5)
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], **L)
    ax.plot([60, 60], [0, 80], **L)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(60 + 10 * np.cos(th), 40 + 10 * np.sin(th), **L)
    for sx, ex in [(0, 18), (102, 120)]:
        ax.plot([sx, ex, ex, sx], [18, 18, 62, 62], **L)
    for sx, ex in [(0, 6), (114, 120)]:
        ax.plot([sx, ex, ex, sx], [30, 30, 50, 50], **L)
    ax.scatter([12, 108], [40, 40], s=6, color=PITCH_LINE, zorder=5)
    ax.plot([0, 0], [36, 44], color="#666", lw=3, zorder=5)
    ax.plot([120, 120], [36, 44], color="#666", lw=3, zorder=5)
    ax.set_xlim(-2, 122)
    ax.set_ylim(82, -2)          # StatsBomb origin is top-left
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _blur(a, sigma):
    """Separable Gaussian blur — avoids a scipy dependency."""
    if sigma <= 0:
        return a
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    f = lambda m: np.convolve(np.pad(m, r, mode="edge"), k, mode="same")[r:-r]
    return np.apply_along_axis(f, 0, np.apply_along_axis(f, 1, a))


def density(sub, minutes, bins_x, bins_y, sigma, value_col=None):
    """Per-minute density. Weighted by value_col when one is given."""
    if not len(sub):
        return np.zeros((bins_y, bins_x))
    w = None
    if value_col:
        w = sub[value_col].fillna(0).values
    H, _, _ = np.histogram2d(sub["x"].values, sub["y"].values,
                             bins=[bins_x, bins_y],
                             range=[[0, 120], [0, 80]], weights=w)
    return _blur(H, sigma).T / max(minutes, 1e-9)


def heat_figure(win, sub_all, total, minutes, spec, opts, header):
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    fig.patch.set_facecolor(BG)
    draw_pitch(ax)

    val = spec["value"]
    dens = density(win, minutes, opts["bx"], opts["by"], opts["sigma"], val)

    if spec["diverging"]:
        lim = (np.abs(dens).max() if opts["autoscale"]
               else max(np.abs(density(sub_all, total, opts["bx"], opts["by"],
                                       opts["sigma"], val)).max()
                        * opts["vmax_mult"], 1e-9))
        lim = max(lim, 1e-9)
        ax.imshow(dens, extent=[0, 120, 80, 0], origin="upper", cmap=DIVERGE,
                  norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                  alpha=0.9, interpolation="bilinear", zorder=1,
                  aspect="equal")
    else:
        vmax = (max(dens.max(), 1e-9) if opts["autoscale"]
                else max(density(sub_all, total, opts["bx"], opts["by"],
                                 opts["sigma"]).max() * opts["vmax_mult"],
                         1e-9))
        ax.imshow(dens, extent=[0, 120, 80, 0], origin="upper", cmap=HEAT,
                  vmin=0, vmax=vmax, alpha=0.9, interpolation="bilinear",
                  zorder=1, aspect="equal")

    if opts["show_events"] and len(win):
        ax.scatter(win["x"], win["y"], s=18, facecolor="none",
                   edgecolor="#2b2b2b", linewidth=0.7, alpha=0.55, zorder=6)

    ax.set_title(header, fontsize=12, fontweight="bold", pad=10)
    ax.text(1, 79.5, "own goal", fontsize=8, color="#a8a8a8", va="bottom")
    ax.text(119, 79.5, "attacking →", fontsize=8, color="#a8a8a8",
            va="bottom", ha="right")
    fig.tight_layout()
    return fig


def timeline_figure(sub, total, lo, hi, label):
    fig, ax = plt.subplots(figsize=(9.2, 1.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    edges = np.arange(0, total + 1, 1.0)
    counts, _ = np.histogram(sub["t"].dropna().values, bins=edges)
    ax.bar(edges[:-1], counts, width=1.0, align="edge", color="#d5d5d5",
           linewidth=0)
    m = (edges[:-1] >= lo) & (edges[:-1] < hi)
    ax.bar(edges[:-1][m], counts[m], width=1.0, align="edge",
           color="#c0392b", linewidth=0)
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel("elapsed minutes", fontsize=9)
    ax.set_title(f"{label} per minute — selected window in red", fontsize=9.5)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cfcfcf")
    fig.tight_layout()
    return fig


def grid_figure(sub, total, window, spec, opts, team, label):
    edges = np.arange(0.0, total + 1e-9, window)
    if edges[-1] < total - 1e-9:
        edges = np.append(edges, total)
    n = len(edges) - 1
    cols = 3
    rows = int(np.ceil(n / cols))

    val = spec["value"]
    if spec["diverging"]:
        shared = max(np.abs(density(sub, total, opts["bx"], opts["by"],
                                    opts["sigma"], val)).max()
                     * opts["vmax_mult"], 1e-9)
    else:
        shared = max(density(sub, total, opts["bx"], opts["by"],
                             opts["sigma"]).max() * opts["vmax_mult"], 1e-9)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows),
                             squeeze=False)
    fig.patch.set_facecolor(BG)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i >= n:
            ax.set_visible(False)
            continue
        lo, hi = edges[i], edges[i + 1]
        w = sub[(sub["t"] >= lo) & (sub["t"] < hi)].dropna(subset=["x", "y"])
        draw_pitch(ax)
        dens = density(w, hi - lo, opts["bx"], opts["by"], opts["sigma"], val)
        if spec["diverging"]:
            ax.imshow(dens, extent=[0, 120, 80, 0], origin="upper",
                      cmap=DIVERGE,
                      norm=TwoSlopeNorm(vmin=-shared, vcenter=0.0,
                                        vmax=shared),
                      alpha=0.9, interpolation="bilinear", zorder=1,
                      aspect="equal")
        else:
            ax.imshow(dens, extent=[0, 120, 80, 0], origin="upper", cmap=HEAT,
                      vmin=0, vmax=shared, alpha=0.9,
                      interpolation="bilinear", zorder=1, aspect="equal")
        if opts["show_events"] and len(w):
            ax.scatter(w["x"], w["y"], s=11, facecolor="none",
                       edgecolor="#2b2b2b", linewidth=0.6, alpha=0.5, zorder=6)
        extra = ""
        if val:
            extra = f"   {w[val].fillna(0).sum():+.2f}"
        ax.set_title(f"{lo:.0f}′–{hi:.0f}′   {len(w)} ev{extra}",
                     fontsize=9.5, fontweight="bold")

    fig.suptitle(f"{team} — {label}, {window:.0f}-minute windows, "
                 f"common colour scale",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.4)
    return fig



# ----------------------------------------------------------------------------
# STREAMLIT VERSION COMPATIBILITY
# ----------------------------------------------------------------------------
def stretch_button(col, label):
    """Full-width button across Streamlit versions."""
    try:
        return col.button(label, width="stretch")
    except TypeError:
        return col.button(label, use_container_width=True)


def show_fig(fig):
    """st.pyplot across versions: `width=` is current, `use_container_width=`
    is deprecated after 2025-12-31 but needed on older builds."""
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)
    finally:
        plt.close(fig)


def show_df(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


# ----------------------------------------------------------------------------
# SIGN IN
# ----------------------------------------------------------------------------
def sign_in():
    """
    Email and password fields in the sidebar, verified against the API before
    anything else loads.

    Env vars and .streamlit/secrets.toml only PREFILL the fields — they never
    bypass them, so a shared deployment always shows who is signed in and lets
    a different account be used. Nothing is written to disk.
    """
    pre_user = os.environ.get("SB_USER", "")
    pre_pw = os.environ.get("SB_PASS", "")
    if not (pre_user and pre_pw):
        try:
            pre_user = pre_user or st.secrets.get("SB_USER", "")
            pre_pw = pre_pw or st.secrets.get("SB_PASS", "")
        except Exception:
            pass

    st.sidebar.header("StatsBomb sign in")

    if st.session_state.get("authed"):
        st.sidebar.success(f"Signed in as {st.session_state['sb_user']}")
        if st.sidebar.button("Sign out"):
            for k in ["authed", "sb_user", "sb_pw"]:
                st.session_state.pop(k, None)
            st.cache_data.clear()
            st.rerun()
        return st.session_state["sb_user"], st.session_state["sb_pw"]

    with st.sidebar.form("signin"):
        email = st.text_input("Email", value=pre_user,
                              placeholder="you@club.com")
        password = st.text_input("Password", value=pre_pw, type="password")
        submitted = st.form_submit_button("Connect")

    if submitted:
        if not (email and password):
            st.sidebar.error("Enter both an email and a password.")
        else:
            try:
                _get(f"{API_BASE}/{COMPETITIONS_VERSION}/competitions",
                     email, password, tries=1)
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else None
                if code in (401, 403):
                    st.sidebar.error("Rejected — check the email and password, "
                                     "and that the account has API access.")
                else:
                    st.sidebar.error(f"API returned {code}. Try again shortly.")
            except Exception as exc:
                st.sidebar.error(f"Could not reach the API: {exc}")
            else:
                st.session_state["authed"] = True
                st.session_state["sb_user"] = email
                st.session_state["sb_pw"] = password
                st.rerun()

    return None, None


# ----------------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Match Heatmap Explorer", layout="wide")
st.title("Match Heatmap Explorer")

user, pw = sign_in()
if not (user and pw):
    st.info("Sign in with your StatsBomb email and password in the sidebar "
            "to begin.")
    st.caption("Credentials are held for this browser session only and are "
               "not stored.")
    st.stop()

with st.sidebar:
    st.header("Match")
    try:
        comps = load_competitions(user, pw)
    except Exception as exc:
        st.error(f"Could not load competitions: {exc}")
        st.stop()

    comps["label"] = (comps["competition_name"] + " — "
                      + comps["season_name"].astype(str))
    comp_label = st.selectbox("Competition and season",
                              sorted(comps["label"].unique()))
    row = comps[comps["label"] == comp_label].iloc[0]

    try:
        matches = load_matches(int(row["competition_id"]),
                               int(row["season_id"]), user, pw)
    except Exception as exc:
        st.error(f"Could not load fixtures: {exc}")
        st.stop()

    date_col = "match_date" if "match_date" in matches.columns else None
    if date_col:
        matches = matches.sort_values(date_col, ascending=False)
    matches["label"] = (
        matches.get(date_col, pd.Series([""] * len(matches))).astype(str)
        + "  " + matches["home_team.home_team_name"].astype(str)
        + " v " + matches["away_team.away_team_name"].astype(str))
    match_label = st.selectbox("Fixture", matches["label"].tolist())
    match_id = int(matches[matches["label"] == match_label]["match_id"].iloc[0])

try:
    ev, total, durs = get_match_data(match_id, user, pw)
except Exception as exc:
    st.error(f"Could not load events for match {match_id}: {exc}")
    st.stop()
teams = sorted(ev["team.name"].dropna().unique().tolist())
if len(teams) < 2:
    st.error(f"Expected two teams, found: {teams}")
    st.stop()

with st.sidebar:
    st.header("View")
    team = st.radio("Team", teams, help="The two teams are in opposite "
                                        "coordinate frames, so they can't be "
                                        "overlaid.")
    metric = st.selectbox("Metric", list(METRICS), index=0)
    spec = METRICS[metric]
    st.caption(spec["help"])

    st.header("Window")
    length = st.slider("Window length (minutes)", MIN_WINDOW,
                       float(round(total)), DEFAULT_WINDOW, step=0.5,
                       help=f"Minimum {MIN_WINDOW:.0f} minutes.")
    max_start = max(0.0, total - length)
    # Streamlit warns if a widget has BOTH a default value and a session_state
    # entry under its key. So seed session_state, clamp it, and let the widget
    # read from it — no `value=` argument.
    if "start" not in st.session_state:
        st.session_state.start = 0.0
    st.session_state.start = float(
        min(max(st.session_state.start, 0.0), max_start))

    c1, c2, c3 = st.columns(3)
    if stretch_button(c1, "◀ −1"):
        st.session_state.start = max(0.0, st.session_state.start - 1.0)
    if stretch_button(c2, "reset"):
        st.session_state.start = 0.0
    if stretch_button(c3, "+1 ▶"):
        st.session_state.start = min(max_start, st.session_state.start + 1.0)

    start = st.slider("Window start (minutes)", 0.0, max(max_start, 0.5),
                      step=0.5, key="start")
    lo, hi = float(start), float(min(start + length, total))

    st.header("Display")
    grid_mode = st.checkbox("Show grid of consecutive windows", value=False)
    show_events = st.checkbox("Show individual events", value=True,
                              help="Leave this on. It shows how many events "
                                   "the shading actually rests on.")
    autoscale = st.checkbox("Scale colour to window", value=False,
                            help="Off = fixed scale, so windows are "
                                 "comparable. On = each window scaled to "
                                 "itself, which exaggerates quiet spells.")
    with st.expander("Smoothing"):
        bx = st.slider("Bins across (length)", 12, 60, 30, step=2)
        by = st.slider("Bins across (width)", 8, 40, 20, step=2)
        sigma = st.slider("Blur (bins)", 0.0, 4.0, 1.6, step=0.2)
        vmax_mult = st.slider("Colour ceiling (× match mean)", 1.0, 6.0, 2.5,
                              step=0.5)

opts = dict(bx=bx, by=by, sigma=sigma, autoscale=autoscale,
            show_events=show_events, vmax_mult=vmax_mult)

sub_all = ev[ev["team.name"] == team]
if spec["types"]:
    sub_all = sub_all[sub_all["type.name"].isin(spec["types"])]
if spec["value"]:
    sub_all = sub_all[sub_all[spec["value"]].notna()]
sub_all = sub_all.dropna(subset=["x", "y"])

win = sub_all[(sub_all["t"] >= lo) & (sub_all["t"] < hi)]
minutes = max(hi - lo, MIN_WINDOW)

# ---- headline numbers
m1, m2, m3, m4 = st.columns(4)
m1.metric("Window", f"{lo:.0f}′ – {hi:.0f}′", f"{hi - lo:.1f} min")
m2.metric("Events in window", f"{len(win):,}",
          f"{len(win) / minutes:.1f} per min")
m3.metric("Match total", f"{len(sub_all):,}",
          f"{len(sub_all) / max(total, 1e-9):.1f} per min")
if spec["value"]:
    m4.metric(f"{metric} in window",
              f"{win[spec['value']].fillna(0).sum():+.2f}",
              f"{win[spec['value']].fillna(0).sum() / minutes:+.3f} per min")
else:
    share = 100 * len(win) / max(len(sub_all), 1)
    m4.metric("Share of match", f"{share:.0f}%")

if len(win) < 10:
    st.warning(f"Only {len(win)} events in this window. The heatmap is a few "
               f"points with a blur applied — read the markers, not the "
               f"shading.")

label = metric
if grid_mode:
    show_fig(grid_figure(sub_all, total, length, spec, opts, team,
                             label))
else:
    left, right = st.columns([3, 1])
    with left:
        header = (f"{team} — {label}\n{lo:.0f}′ to {hi:.0f}′   "
                  f"({hi - lo:.1f} min, {len(win)} events, "
                  f"{len(win) / minutes:.1f}/min)")
        show_fig(heat_figure(win, sub_all, total, minutes, spec, opts,
                             header))
        show_fig(timeline_figure(sub_all, total, lo, hi, label))
    with right:
        st.subheader("Top zones")
        if len(win):
            thirds = pd.cut(win["x"], [0, 40, 80, 120],
                            labels=["Defensive", "Middle", "Final"])
            by_third = win.groupby(thirds, observed=False).size()
            show_df(by_third.rename("events").to_frame())
            if spec["value"]:
                v = win.groupby(thirds, observed=False)[spec["value"]].sum()
                show_df(v.round(3).rename(metric).to_frame())
        else:
            st.caption("No events in this window.")

        st.subheader("Players")
        if len(win) and "player.name" in win.columns:
            agg = (win.groupby("player.name").size()
                   .sort_values(ascending=False).head(10))
            show_df(agg.rename("events").to_frame())

with st.expander("Match and data notes"):
    st.write(f"**Match {match_id}** — {match_label}")
    st.write(f"Playing time from events: **{total:.1f} min** "
             f"({', '.join(f'P{int(p)}: {v:.1f}' for p, v in durs.items())}). "
             f"Built per period, because `minute` resets to 45 at half-time "
             f"and a raw clock would drop first-half stoppage.")
    st.write(f"Events for {team} on this metric: **{len(sub_all):,}**")
    if spec["value"]:
        st.write("OBV is summed rather than counted and can be negative, so "
                 "the colour scale is diverging and centred on zero.")
    st.write("Density is per minute, so window length does not change "
             "intensity. Coordinates are normalised so the acting team "
             "attacks toward x = 120.")
