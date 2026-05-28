
import sqlite3
import hashlib

DB = "users.db"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        password TEXT,
        role TEXT
    )
    ''')

    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    hashed = hash_password(password)

    cur.execute(
        "INSERT INTO users VALUES(?,?,?)",
        (username, hashed, "user")
    )

    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
        "SELECT password FROM users WHERE username=?",
        (username,)
    )

    row = cur.fetchone()

    conn.close()

    if not row:
        return False

    return row[0] == hash_password(password)
