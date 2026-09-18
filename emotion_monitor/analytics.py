import io
import base64
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Color palette for each emotion
EMOTION_COLORS = {
    "Angry": "#e74c3c",
    "Disgusted": "#8e44ad",
    "Fearful": "#34495e",
    "Happy": "#f1c40f",
    "Neutral": "#95a5a6",
    "Sad": "#3498db",
    "Surprised": "#e67e22",
}

EMOTION_ORDER = ["Angry", "Disgusted", "Fearful", "Happy",
                 "Neutral", "Sad", "Surprised"]


# ============================================================== #
#                     BASIC CHART FUNCTIONS                      #
# ============================================================== #

def create_pie_chart(emotion_counts):
    """Create a pie chart from emotion counts. Returns base64 PNG string."""
    labels, sizes = [], []
    for k, v in emotion_counts.items():
        if v > 0:
            labels.append(k)
            sizes.append(v)

    if not labels:
        labels, sizes = ["Neutral"], [1]

    colors = [EMOTION_COLORS.get(l, "#cccccc") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           colors=colors, startangle=140,
           textprops={'fontsize': 11})
    ax.axis('equal')
    ax.set_title('Emotion Distribution', fontsize=14, fontweight='bold')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_timeline_chart(timeline):
    """Create a line chart of emotion over time. Returns base64 PNG or None."""
    if not timeline:
        return None

    y_map = {e: i for i, e in enumerate(EMOTION_ORDER)}

    xs = [p["second"] for p in timeline]
    ys = [y_map.get(p["emotion"], 4) for p in timeline]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, ys, marker='o', linewidth=2, color="#1e3a8a",
            markersize=6, markerfacecolor="#3b82f6")

    # Colored background bands per emotion
    for i, name in enumerate(EMOTION_ORDER):
        ax.axhspan(i - 0.4, i + 0.4,
                   color=EMOTION_COLORS[name], alpha=0.08)

    ax.set_yticks(range(len(EMOTION_ORDER)))
    ax.set_yticklabels(EMOTION_ORDER)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Emotion Timeline", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ============================================================== #
#                     OBSERVATION GENERATION                     #
# ============================================================== #

def generate_observations(timeline, emotion_counts, transitions,
                          avg_confidence, duration):
    """
    Rule-based, carefully-worded engagement observations.
    These are ESTIMATES — not claims about comprehension.
    """
    notes = []

    if duration and duration > 0:
        # Long neutral stretches
        neutral_secs = [p["second"] for p in timeline
                        if p["emotion"] == "Neutral"]
        if len(neutral_secs) > duration * 0.5:
            notes.append(
                f"High neutral-expression period detected "
                f"({len(neutral_secs)}s of {duration}s)."
            )

        # Positive expression windows
        positive = [p["second"] for p in timeline
                    if p["emotion"] in ("Happy", "Surprised")]
        if positive:
            shown = ", ".join(f"{s}s" for s in positive[:3])
            more = " ..." if len(positive) > 3 else ""
            notes.append(
                f"Positive-expression windows observed around {shown}{more}."
            )

        # Variability
        if transitions >= 8:
            notes.append("Expression variability: HIGH — many transitions.")
        elif transitions >= 4:
            notes.append("Expression variability: MODERATE.")
        else:
            notes.append("Expression variability: LOW — stable expression.")

        # Confidence note
        if avg_confidence < 0.5:
            notes.append(
                f"Average prediction confidence is low "
                f"({avg_confidence * 100:.1f}%). "
                f"Interpret results with caution."
            )
        else:
            notes.append(
                f"Average prediction confidence: "
                f"{avg_confidence * 100:.1f}% (reasonable reliability)."
            )

    if not notes:
        notes.append("No significant expression-pattern changes detected.")

    notes.append(
        "Disclaimer: Facial expressions alone cannot establish attention "
        "or comprehension. These are estimated engagement indicators, "
        "not facts about the student's internal state."
    )

    return "\n".join(f"- {n}" for n in notes)


# ============================================================== #
#                     PLAIN TEXT REPORT                          #
# ============================================================== #

def build_report_text(session_row):
    """Build a plain-text downloadable report from a session DB row."""
    lines = [
        "=" * 60,
        "  EDUSENSE - SESSION REPORT",
        "=" * 60,
        f"Session ID       : {session_row['id']}",
    ]

    try:
        username = session_row['username']
    except (IndexError, KeyError):
        username = 'N/A'

    lines += [
        f"Student          : {username}",
        f"Date             : {session_row['created_at']}",
        f"Duration         : {session_row['duration']} seconds",
        f"Frames analysed  : {session_row['total_faces']}",
        f"Dominant emotion : {session_row['dominant_emotion']}",
        f"Avg confidence   : {round((session_row['avg_confidence'] or 0) * 100, 1)}%",
        f"Transitions      : {session_row['transitions']}",
        "",
        "-" * 60,
        "EMOTION DISTRIBUTION",
        "-" * 60,
    ]

    counts = json.loads(session_row['emotion_counts'] or "{}")
    total = sum(counts.values()) or 1
    for k in EMOTION_ORDER:
        v = counts.get(k, 0)
        if v > 0:
            lines.append(f"  {k:<12} {v:>5}   ({v / total * 100:5.1f}%)")

    lines += [
        "",
        "-" * 60,
        "OBSERVATIONS (estimates, not comprehension facts)",
        "-" * 60,
        session_row['observations'] or "N/A",
        "",
        "=" * 60,
        "Generated by EduSense - Student Emotion Monitoring System",
        "=" * 60,
    ]

    return "\n".join(lines)


# ============================================================== #
#                 CLASS-WIDE ANALYTICS (ADMIN)                   #
# ============================================================== #

def create_admin_overview_charts(rows):
    """
    Build class-wide charts for the admin panel.

    `rows` is a list of sqlite3.Row with columns:
        created_at, dominant_emotion, duration, avg_confidence

    Returns dict of base64 PNG strings:
        {'dominant_bar': ..., 'daily_line': ..., 'hourly_heat': ...}
    """
    charts = {}

    # ---------- 1) Bar: dominant emotion across all sessions ---------- #
    dom_counts = {e: 0 for e in EMOTION_ORDER}
    for r in rows:
        if r['dominant_emotion'] in dom_counts:
            dom_counts[r['dominant_emotion']] += 1

    labels = [e for e in EMOTION_ORDER if dom_counts[e] > 0]
    values = [dom_counts[e] for e in labels]

    if not labels:
        labels, values = ["Neutral"], [1]

    colors_bar = [EMOTION_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=colors_bar)
    ax.set_title("Dominant Emotion Across All Sessions",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of sessions")
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, str(v), ha='center', fontsize=10)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    charts['dominant_bar'] = base64.b64encode(buf.read()).decode('utf-8')

    # ---------- 2) Line: sessions per day (last 14 days) ---------- #
    day_counts = {}
    for r in rows:
        try:
            day = str(r['created_at']).split(' ')[0]
            day_counts[day] = day_counts.get(day, 0) + 1
        except Exception:
            pass

    sorted_days = sorted(day_counts.keys())[-14:]
    day_values = [day_counts[d] for d in sorted_days]

    if not sorted_days:
        import datetime as _dt
        sorted_days = [_dt.datetime.now().strftime('%Y-%m-%d')]
        day_values = [0]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(sorted_days, day_values, marker='o', linewidth=2,
            color="#1e3a8a", markerfacecolor="#3b82f6")
    ax.set_title("Sessions Per Day", fontsize=13, fontweight='bold')
    ax.set_ylabel("Sessions")
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    charts['daily_line'] = base64.b64encode(buf.read()).decode('utf-8')

    # ---------- 3) Heatmap: hour of day vs dominant emotion ---------- #
    heatmap = {e: [0] * 24 for e in EMOTION_ORDER}
    for r in rows:
        try:
            ts = str(r['created_at'])
            hour = int(ts.split(' ')[1].split(':')[0])
            if r['dominant_emotion'] in heatmap:
                heatmap[r['dominant_emotion']][hour] += 1
        except Exception:
            pass

    matrix = np.array([heatmap[e] for e in EMOTION_ORDER])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(matrix, aspect='auto', cmap='Blues')
    ax.set_yticks(range(len(EMOTION_ORDER)))
    ax.set_yticklabels(EMOTION_ORDER, fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=8)
    ax.set_xlabel("Hour of day")
    ax.set_title("Emotion Peaks by Hour", fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Sessions')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    charts['hourly_heat'] = base64.b64encode(buf.read()).decode('utf-8')

    return charts


def compute_class_stats(rows):
    """Aggregate KPIs for admin dashboard."""
    total = len(rows)
    if total == 0:
        return {
            'total_sessions': 0,
            'positive_pct': 0,
            'negative_pct': 0,
            'neutral_pct': 0,
            'avg_confidence_pct': 0,
            'avg_duration': 0,
        }

    positive = sum(1 for r in rows
                   if r['dominant_emotion'] in ('Happy', 'Surprised'))
    negative = sum(1 for r in rows
                   if r['dominant_emotion'] in ('Angry', 'Disgusted',
                                                'Fearful', 'Sad'))
    neutral = sum(1 for r in rows if r['dominant_emotion'] == 'Neutral')

    avg_conf = sum((r['avg_confidence'] or 0) for r in rows) / total
    avg_dur = sum(r['duration'] for r in rows) / total

    return {
        'total_sessions': total,
        'positive_pct': round(positive / total * 100, 1),
        'negative_pct': round(negative / total * 100, 1),
        'neutral_pct': round(neutral / total * 100, 1),
        'avg_confidence_pct': round(avg_conf * 100, 1),
        'avg_duration': round(avg_dur, 1),
    }
