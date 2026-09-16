import sqlite3
import bcrypt


def create_database():
    conn = sqlite3.connect("users.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    
def create_user(name, email, password):
  # password to bytes
  password_bytes = password.encode("utf-8")
  
  # hash password
  password_hash = bcrypt.hashpw(
    password_bytes,
    bcrypt.gensalt()
)
  
  conn = sqlite3.connect("users.db")
  cursor = conn.cursor()
  
  try :
    cursor.execute(
      """
      INSERT INTO users (name, email, password_hash)
      VALUES (?, ?, ?)
      """,
      (name, email, password_hash.decode("utf-8"))
    )
    
    conn.commit()
    
    return True
  
  except sqlite3.IntregrityError:
    # email exited
    return False
  
  finally:
    conn.close()

def login_user(email , password):
  conn = sqlite3.connect("user.db")
  cursor = conn.cursor()
  
  cursor.execute(
  """
  SELECT name , password_hash
  FROM users
  WHERE email = ?
  """,
  (email , )
  )
  
  user = cursor.fetchone()
  
  conn.close()
  
  if user is None:
    return None
  
  name , password_hash = user
  
  if bcrypt.checkpw(
    password.encode("utf-8"),
    password_hash("utf-8")
  ):
    return name
  
  return None
      
      
create_database()