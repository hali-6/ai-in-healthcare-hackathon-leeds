import database_aihack as db
from user_data_storage import init_user_data_db

db.init_db()
init_user_data_db()

print("Database initialised successfully!")