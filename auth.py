# modules/auth.py
# =============================================================
# Authentication — SQLite + bcrypt hashing + session tokens
# =============================================================
import sqlite3, hashlib, os, datetime, json, secrets
from typing import Optional

DB_PATH = "trading_users.db"

# ── DB INIT ──────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            role      TEXT DEFAULT 'user',
            status    TEXT DEFAULT 'pending',
            plan      TEXT DEFAULT 'Free',
            expiry    TEXT DEFAULT '2099-12-31',
            email     TEXT DEFAULT '',
            phone     TEXT DEFAULT '',
            exam      TEXT DEFAULT '',
            txn_id    TEXT DEFAULT '',
            joined    TEXT DEFAULT '',
            last_login TEXT DEFAULT ''
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token     TEXT PRIMARY KEY,
            username  TEXT NOT NULL,
            created   TEXT NOT NULL,
            expires   TEXT NOT NULL
        )
    """)
    # Default admin
    _admin_hash = _hash("admin123")
    cur.execute("""
        INSERT OR IGNORE INTO users
        (username,password,role,status,plan,expiry,joined)
        VALUES (?,?,?,?,?,?,?)
    """, ("admin", _admin_hash, "admin", "active", "Premium",
          "2099-12-31", str(datetime.date.today())))
    con.commit(); con.close()


# ── HASHING ──────────────────────────────────────────────────
def _hash(password: str) -> str:
    """SHA-256 hash with salt — bcrypt preferred if available."""
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        salt = "trading_pro_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()

def _verify(password: str, hashed: str) -> bool:
    try:
        import bcrypt
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ImportError:
        salt = "trading_pro_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest() == hashed


# ── SESSION TOKENS ────────────────────────────────────────────
def create_session(username: str) -> str:
    token = secrets.token_hex(32)
    expires = (datetime.datetime.now() + datetime.timedelta(hours=24)).isoformat()
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?)",
                (token, username, datetime.datetime.now().isoformat(), expires))
    con.commit(); con.close()
    return token

def validate_session(token: str) -> Optional[str]:
    if not token: return None
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT username, expires FROM sessions WHERE token=?", (token,)
    ).fetchone()
    con.close()
    if not row: return None
    if datetime.datetime.now() > datetime.datetime.fromisoformat(row[1]):
        return None  # expired
    return row[0]

def revoke_session(token: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM sessions WHERE token=?", (token,))
    con.commit(); con.close()


# ── USER CRUD ─────────────────────────────────────────────────
def login(username: str, password: str) -> dict:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT * FROM users WHERE username=?", (username,)
    ).fetchone()
    con.close()
    if not row: return {"ok": False, "error": "Invalid username or password"}

    cols = ["id","username","password","role","status","plan","expiry",
            "email","phone","exam","txn_id","joined","last_login"]
    user = dict(zip(cols, row))

    if not _verify(password, user["password"]):
        return {"ok": False, "error": "Invalid username or password"}
    if user["status"] != "active":
        return {"ok": False, "error": "Account pending admin approval"}
    try:
        if (user["plan"] != "Free" and
                datetime.date.today() >
                datetime.datetime.strptime(user["expiry"],"%Y-%m-%d").date()):
            return {"ok": False, "error": "Subscription expired"}
    except Exception:
        pass

    # Update last login
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE users SET last_login=? WHERE username=?",
                (str(datetime.datetime.now())[:19], username))
    con.commit(); con.close()

    token = create_session(username)
    return {"ok": True, "token": token, "user": user}

def signup(username: str, password: str, email: str = "",
           phone: str = "", **kwargs) -> dict:
    if len(password) < 6:
        return {"ok": False, "error": "Password must be 6+ characters"}
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("""
            INSERT INTO users (username,password,email,phone,joined,status)
            VALUES (?,?,?,?,?,?)
        """, (username, _hash(password), email, phone,
              str(datetime.date.today()), "active"))
        con.commit()
        return {"ok": True}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": f"Username '{username}' already taken"}
    finally:
        con.close()

def get_user(username: str) -> Optional[dict]:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    con.close()
    if not row: return None
    cols = ["id","username","password","role","status","plan","expiry",
            "email","phone","exam","txn_id","joined","last_login"]
    return dict(zip(cols, row))

def get_all_users() -> list:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT * FROM users WHERE username != 'admin'").fetchall()
    con.close()
    cols = ["id","username","password","role","status","plan","expiry",
            "email","phone","exam","txn_id","joined","last_login"]
    return [dict(zip(cols, r)) for r in rows]

def update_user(username: str, **fields):
    if not fields: return
    safe = {k:v for k,v in fields.items()
            if k in ["status","plan","expiry","email","phone","txn_id","role"]}
    if not safe: return
    sets = ", ".join(f"{k}=?" for k in safe)
    vals = list(safe.values()) + [username]
    con = sqlite3.connect(DB_PATH)
    con.execute(f"UPDATE users SET {sets} WHERE username=?", vals)
    con.commit(); con.close()

def delete_user(username: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM users WHERE username=?", (username,))
    con.commit(); con.close()

def create_user_admin(username: str, password: str, plan: str,
                      days: int, email: str, phone: str, txn_id: str) -> dict:
    expiry = str(datetime.date.today() + datetime.timedelta(days=days))
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("""
            INSERT INTO users
            (username,password,role,status,plan,expiry,email,phone,txn_id,joined)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (username, _hash(password), "user", "active", plan,
              expiry, email, phone, txn_id, str(datetime.date.today())))
        con.commit()
        return {"ok": True, "expiry": expiry}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Username already exists"}
    finally:
        con.close()


# ── MIGRATE FROM users.json ───────────────────────────────────
def migrate_json(json_path: str = "users.json"):
    """One-time migration from users.json to SQLite."""
    if not os.path.exists(json_path): return 0
    try:
        users = json.load(open(json_path))
        count = 0
        for uname, udata in users.items():
            if uname == "admin": continue
            old_pw = udata.get("password","pass123")
            # Already hashed? Check length
            if len(old_pw) == 64:  # SHA256 hex
                new_pw = old_pw
            else:
                new_pw = _hash(old_pw)
            con = sqlite3.connect(DB_PATH)
            try:
                con.execute("""
                    INSERT OR IGNORE INTO users
                    (username,password,role,status,plan,expiry,email,phone,joined)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (uname, new_pw,
                      udata.get("role","user"),
                      udata.get("status","active"),
                      udata.get("plan","Free"),
                      udata.get("expiry","2099-12-31"),
                      udata.get("email",""),
                      udata.get("phone",""),
                      udata.get("joined", str(datetime.date.today()))))
                con.commit(); count += 1
            except Exception: pass
            finally: con.close()
        return count
    except Exception as e:
        return 0
