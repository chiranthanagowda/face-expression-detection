import os
import sqlite3
import base64
import json
import time as _time
import threading
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, g, send_file, Response, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash

import matplotlib
matplotlib.use('Agg')

# ---- Our modules ----
from TestEmotionDetector import (run_emotion_detection,
                                  start_session, get_frame, get_status,
                                  stop_session, is_session_active,
                                  get_final_result)
from analytics import (create_pie_chart, create_timeline_chart,
                       generate_observations, build_report_text,
                       create_admin_overview_charts, compute_class_stats)
from pdf_report import build_session_pdf


# ----------------------------- App Config ----------------------------- #
app = Flask(__name__)
app.secret_key = 'change-this-to-a-secret-key-in-production'

DATABASE = 'database.db'
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---- Finish cache: prevents double-save on duplicate /predict_finish calls ----
_finish_cache = {
    'url': None,
    'expires': 0,
}
_finish_lock = threading.Lock()


# ----------------------------- Database ----------------------------- #
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT DEFAULT 'student',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                duration INTEGER NOT NULL,
                total_frames INTEGER NOT NULL,
                total_faces INTEGER NOT NULL,
                dominant_emotion TEXT NOT NULL,
                avg_confidence REAL DEFAULT 0,
                transitions INTEGER DEFAULT 0,
                timeline TEXT,
                emotion_counts TEXT,
                chart BLOB,
                timeline_chart BLOB,
                observations TEXT,
                lecture_markers TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lecture_markers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                seconds INTEGER NOT NULL,
                label TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')

        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
                ('admin', 'admin@system.com',
                 generate_password_hash('admin123'), 'admin')
            )
        db.commit()


# ----------------------------- Decorators ----------------------------- #
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get('role') != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return wrapper


# ----------------------------- Basic Routes ----------------------------- #
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']

        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))

        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT id FROM users WHERE username=? OR email=?",
                       (username, email))
        if cursor.fetchone():
            flash('Username or email already exists.', 'danger')
            return redirect(url_for('register'))

        cursor.execute(
            "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
            (username, email, generate_password_hash(password), 'student')
        )
        db.commit()
        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash(f"Welcome back, {user['username']}!", 'success')
            if user['role'] == 'admin':
                return redirect(url_for('admin'))
            return redirect(url_for('dashboard'))

        flash('Invalid credentials.', 'danger')
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))


# ----------------------------- Dashboard ----------------------------- #
@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT id, duration, total_faces, dominant_emotion, avg_confidence, "
        "transitions, chart, created_at "
        "FROM sessions WHERE user_id=? ORDER BY id DESC",
        (session['user_id'],)
    )
    rows = cursor.fetchall()

    sessions_list = []
    for row in rows:
        chart_b64 = base64.b64encode(row['chart']).decode('utf-8') if row['chart'] else None
        sessions_list.append({
            'id': row['id'],
            'duration': row['duration'],
            'total_faces': row['total_faces'],
            'dominant_emotion': row['dominant_emotion'],
            'avg_confidence': round((row['avg_confidence'] or 0) * 100, 1),
            'transitions': row['transitions'],
            'chart': chart_b64,
            'created_at': row['created_at']
        })

    return render_template('dashboard.html', sessions=sessions_list)


# ----------------------------- PREDICT (OpenCV window) ----------------------------- #
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            duration = int(request.form.get('duration', 30))
        except ValueError:
            duration = 30

        if duration < 5 or duration > 600:
            flash('Duration must be between 5 and 600 seconds.', 'warning')
            return redirect(url_for('predict'))

        marker_raw = request.form.get('lecture_markers', '').strip()
        markers = []
        if marker_raw:
            for chunk in marker_raw.split(','):
                if ':' in chunk:
                    sec_str, label = chunk.split(':', 1)
                    try:
                        markers.append((int(sec_str.strip()), label.strip()))
                    except ValueError:
                        pass

        flash(f'Camera starting for {duration}s. Press "q" on the camera window to stop early.', 'info')

        result = run_emotion_detection(duration)

        if result['total_faces'] == 0:
            flash('No faces detected during the session.', 'warning')
            return redirect(url_for('predict'))

        pie_b64 = create_pie_chart(result['emotion_counts'])
        tl_b64 = create_timeline_chart(result['timeline']) or ''
        pie_bytes = base64.b64decode(pie_b64)
        tl_bytes = base64.b64decode(tl_b64) if tl_b64 else None

        observations = generate_observations(
            result['timeline'],
            result['emotion_counts'],
            result['transitions'],
            result['avg_confidence'],
            duration
        )

        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO sessions
              (user_id, duration, total_frames, total_faces,
               dominant_emotion, avg_confidence, transitions,
               timeline, emotion_counts, chart, timeline_chart,
               observations, lecture_markers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['user_id'], duration,
            result['total_faces'], result['total_faces'],
            result['dominant_emotion'], result['avg_confidence'],
            result['transitions'],
            json.dumps(result['timeline']),
            json.dumps(result['emotion_counts']),
            pie_bytes, tl_bytes,
            observations,
            json.dumps(markers)
        ))
        db.commit()
        session_id = cursor.lastrowid

        for sec, label in markers:
            cursor.execute(
                "INSERT INTO lecture_markers (session_id, seconds, label) VALUES (?, ?, ?)",
                (session_id, sec, label)
            )
        db.commit()

        return redirect(url_for('result', session_id=session_id))

    return render_template('predict.html')


# ----------------------------- PREDICT LIVE (browser webcam) ----------------------------- #
@app.route('/predict_live', methods=['POST'])
@login_required
def predict_live():
    try:
        duration = int(request.form.get('duration', 30))
    except ValueError:
        duration = 30

    if duration < 5 or duration > 600:
        flash('Duration must be between 5 and 600 seconds.', 'warning')
        return redirect(url_for('predict'))

    if is_session_active():
        flash('A session is already running. Please wait.', 'warning')
        return redirect(url_for('predict'))

    session['pending_markers'] = request.form.get('lecture_markers', '').strip()

    start_session(duration, user_id=session['user_id'])
    return redirect(url_for('predict_live_view', duration=duration))


@app.route('/predict_live/view')
@login_required
def predict_live_view():
    duration = int(request.args.get('duration', 30))
    return render_template('predict_live.html', duration=duration)


@app.route('/video_feed')
@login_required
def video_feed():
    """MJPEG stream endpoint."""
    def generate():
        import time as _t
        while True:
            frame_bytes = get_frame()
            if frame_bytes is None:
                _t.sleep(0.03)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')
            _t.sleep(0.03)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_status')
@login_required
def predict_status():
    return jsonify(get_status())


@app.route('/predict_stop', methods=['POST'])
@login_required
def predict_stop():
    stop_session()
    return jsonify({'ok': True})


@app.route('/predict_finish')
@login_required
def predict_finish():
    """
    Save results once — duplicate calls (from JS re-firing) return the
    cached redirect URL instead of running the whole pipeline again.
    """
    global _finish_cache

    # ---- 1. Check cache first ----
    with _finish_lock:
        now = _time.time()
        if _finish_cache['url'] and now < _finish_cache['expires']:
            cached_url = _finish_cache['url']
            print(f"[FINISH] Cache hit — redirecting to {cached_url}")
            return redirect(cached_url)

    # ---- 2. Grab final result ----
    result = get_final_result()

    # If None (already popped by a previous call), wait briefly and retry
    if result is None:
        print("[FINISH] Result None — waiting 1.5s for worker to settle...")
        _time.sleep(1.5)
        result = get_final_result()

    if result is None:
        print("[FINISH] No result available (already consumed).")
        flash('Session data unavailable — please start a new session.',
              'warning')
        return redirect(url_for('predict'))

    if result['total_faces'] == 0:
        print("[FINISH] Zero faces — cannot save session.")
        flash('No faces detected during the session.', 'warning')
        return redirect(url_for('predict'))

    # ---- 3. Build charts ----
    pie_b64 = create_pie_chart(result['emotion_counts'])
    tl_b64 = create_timeline_chart(result['timeline']) or ''
    pie_bytes = base64.b64decode(pie_b64)
    tl_bytes = base64.b64decode(tl_b64) if tl_b64 else None

    # ---- 4. Observations ----
    observations = generate_observations(
        result['timeline'], result['emotion_counts'],
        result['transitions'], result['avg_confidence'],
        result['duration']
    )

    # ---- 5. Markers ----
    marker_raw = session.pop('pending_markers', '')
    markers = []
    if marker_raw:
        for chunk in marker_raw.split(','):
            if ':' in chunk:
                sec_str, label = chunk.split(':', 1)
                try:
                    markers.append((int(sec_str.strip()), label.strip()))
                except ValueError:
                    pass

    # ---- 6. Save to DB ----
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        INSERT INTO sessions
          (user_id, duration, total_frames, total_faces,
           dominant_emotion, avg_confidence, transitions,
           timeline, emotion_counts, chart, timeline_chart,
           observations, lecture_markers)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session['user_id'], result['duration'],
        result['total_faces'], result['total_faces'],
        result['dominant_emotion'], result['avg_confidence'],
        result['transitions'],
        json.dumps(result['timeline']),
        json.dumps(result['emotion_counts']),
        pie_bytes, tl_bytes,
        observations,
        json.dumps(markers)
    ))
    db.commit()
    session_id = cursor.lastrowid

    for sec, label in markers:
        cursor.execute(
            "INSERT INTO lecture_markers (session_id, seconds, label) VALUES (?, ?, ?)",
            (session_id, sec, label)
        )
    db.commit()

    # ---- 7. Cache the redirect URL for 60 seconds ----
    result_url = url_for('result', session_id=session_id)
    with _finish_lock:
        _finish_cache['url'] = result_url
        _finish_cache['expires'] = _time.time() + 60

    print(f"[FINISH] Saved session #{session_id} — redirecting to {result_url}")
    return redirect(result_url)


# ----------------------------- RESULT ----------------------------- #
@app.route('/result/<int:session_id>')
@login_required
def result(session_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    row = cursor.fetchone()

    if not row:
        flash('Session not found.', 'danger')
        return redirect(url_for('dashboard'))

    if row['user_id'] != session['user_id'] and session.get('role') != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))

    pie_b64 = base64.b64encode(row['chart']).decode('utf-8') if row['chart'] else None
    tl_b64 = base64.b64encode(row['timeline_chart']).decode('utf-8') if row['timeline_chart'] else None

    cursor.execute(
        "SELECT seconds, label FROM lecture_markers WHERE session_id=? ORDER BY seconds",
        (session_id,)
    )
    markers = cursor.fetchall()

    per_topic = []
    try:
        timeline_data = json.loads(row['timeline'] or '[]')
        emotion_order = ["Angry", "Disgusted", "Fearful", "Happy",
                         "Neutral", "Sad", "Surprised"]

        if markers and timeline_data:
            marker_list = [(m['seconds'], m['label']) for m in markers]
            boundaries = marker_list + [(row['duration'], '__END__')]
            for i in range(len(boundaries) - 1):
                start_sec, label = boundaries[i]
                end_sec = boundaries[i + 1][0]
                seg = [p for p in timeline_data
                       if start_sec <= p['second'] < end_sec]
                if not seg:
                    continue
                counts = {e: 0 for e in emotion_order}
                for p in seg:
                    if p['emotion'] in counts:
                        counts[p['emotion']] += 1
                total = sum(counts.values()) or 1
                per_topic.append({
                    'label': label,
                    'start': start_sec,
                    'end': end_sec,
                    'counts': {k: round(v / total * 100, 1)
                               for k, v in counts.items()}
                })
    except Exception as e:
        print(f"[WARN] per-topic breakdown failed: {e}")
        per_topic = []

    return render_template('result.html',
                           session_id=row['id'],
                           chart=pie_b64,
                           timeline_chart=tl_b64,
                           duration=row['duration'],
                           total_faces=row['total_faces'],
                           dominant=row['dominant_emotion'],
                           avg_confidence=round((row['avg_confidence'] or 0) * 100, 1),
                           transitions=row['transitions'],
                           observations=row['observations'],
                           created_at=row['created_at'],
                           per_topic=per_topic)


# ----------------------------- PDF / TEXT Report ----------------------------- #
@app.route('/download_report/<int:session_id>')
@login_required
def download_report(session_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT s.*, u.username FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.id = ?
    ''', (session_id,))
    row = cursor.fetchone()

    if not row:
        flash('Session not found.', 'danger')
        return redirect(url_for('dashboard'))

    if row['user_id'] != session['user_id'] and session.get('role') != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))

    try:
        filename = f"session_{session_id}_report.pdf"
        path = os.path.join(REPORTS_DIR, filename)
        build_session_pdf(row, path)
        return send_file(path, as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        text = build_report_text(row)
        filename = f"session_{session_id}_report.txt"
        path = os.path.join(REPORTS_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        flash('PDF generation failed — sent text report instead.', 'warning')
        return send_file(path, as_attachment=True, download_name=filename)


# ----------------------------- Delete Session ----------------------------- #
@app.route('/delete_session/<int:session_id>', methods=['POST'])
@login_required
def delete_session(session_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT user_id FROM sessions WHERE id=?", (session_id,))
    row = cursor.fetchone()

    if not row:
        flash('Session not found.', 'danger')
        return redirect(url_for('dashboard'))

    if row['user_id'] != session['user_id'] and session.get('role') != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))

    cursor.execute("DELETE FROM lecture_markers WHERE session_id=?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    db.commit()
    flash('Session deleted successfully.', 'success')

    if session.get('role') == 'admin':
        return redirect(url_for('admin'))
    return redirect(url_for('dashboard'))


# ----------------------------- ADMIN ----------------------------- #
@app.route('/admin')
@login_required
@admin_required
def admin():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT s.id, s.user_id, s.duration, s.total_faces,
               s.dominant_emotion, s.avg_confidence, s.transitions,
               s.created_at, u.username, u.email
        FROM sessions s JOIN users u ON s.user_id = u.id
        ORDER BY s.id DESC
    ''')
    rows = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) as c FROM users WHERE role='student'")
    total_students = cursor.fetchone()['c']

    charts = create_admin_overview_charts(rows)
    stats = compute_class_stats(rows)

    return render_template('admin.html',
                           sessions=rows,
                           total_students=total_students,
                           stats=stats,
                           charts=charts)


# ----------------------------- Entry Point ----------------------------- #
if __name__ == '__main__':
    init_db()
    app.run(debug=True, use_reloader=False)