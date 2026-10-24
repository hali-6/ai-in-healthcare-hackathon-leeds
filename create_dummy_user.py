import database_aihack as db
import sqlite3
 
# Initialise database
db.init_db()
 
# Dummy login details
username = "testpatient01"
password = "TestPatient2026!"
 
try:
    # Register user
    user_id = db.register_user(username, password)
 
    print("User created successfully!")
    print("User ID:", user_id)
 
except sqlite3.IntegrityError:
    print("User already exists.")
 
# Test login
user = db.authenticate_user(username, password)
 
if user:
    print("Login successful!")
    print("Username:", user["username"])
    print("User ID:", user["user_id"])
else:
    print("Login failed!")
 