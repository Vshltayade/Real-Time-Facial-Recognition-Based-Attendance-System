from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, date
from database import get_connection

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Attendance Backend Running"

# ================= MARK ATTENDANCE =================
@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():
    data = request.json
    name = data.get("name")
    course_id = data.get("course_id")   # ✅ ADDED

    if not name or not course_id:
        return jsonify({"message": "Name or Course missing"}), 400

    today = date.today()
    now = datetime.now().time()

    conn = get_connection()
    cur = conn.cursor()

    # Prevent duplicate attendance per day per course
    cur.execute("""
        SELECT id FROM attendance
        WHERE student_name=%s AND date=%s AND course_id=%s
    """, (name, today, course_id))

    if cur.fetchone():
        cur.close()
        conn.close()
        return jsonify({"message": "Already marked"}), 200

    cur.execute("""
        INSERT INTO attendance (student_name, date, time, course_id)
        VALUES (%s, %s, %s, %s)
    """, (name, today, now, course_id))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": f"Attendance marked for {name}"}), 200

# ================= GET ATTENDANCE =================
@app.route("/attendance", methods=["GET"])
def get_attendance():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT 
            a.student_name,
            a.date,
            a.time,
            c.course_name
        FROM attendance a
        LEFT JOIN courses c ON a.course_id = c.course_id
        ORDER BY a.date DESC, a.time DESC
    """)

    rows = cur.fetchall()

    # 🔥 FIX: convert non-JSON-serializable types
    for row in rows:
        if row["date"]:
            row["date"] = row["date"].strftime("%Y-%m-%d")
        if row["time"]:
            row["time"] = str(row["time"])

    cur.close()
    conn.close()

    return jsonify(rows)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
