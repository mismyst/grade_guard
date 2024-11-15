from flask import Flask, jsonify, request
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# MySQL database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Nayantra1234@'
app.config['MYSQL_DATABASE'] = 'wow'

# Function to establish a database connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DATABASE']
        )
        if conn.is_connected():
            return conn
    except Error as e:
        print("Error connecting to MySQL", e)
        return None

# Home route
@app.route('/')
def home():
    return "Welcome to the Flask MySQL App!"

# Route to retrieve all records from a table
@app.route('/records', methods=['GET'])
def get_records():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT * FROM table")
        records = cursor.fetchall()
        return jsonify(records)
    except Error as e:
        return jsonify({"error": str(e)})
    finally:
        cursor.close()
        conn.close()

# Route to add a new record
@app.route('/add_record', methods=['POST'])
def add_record():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Example of inserting data
        sql = "INSERT INTO your_table (column1, column2) VALUES (%s, %s)"
        values = (data['column1'], data['column2'])
        cursor.execute(sql, values)
        conn.commit()
        return jsonify({"message": "Record added successfully!"})
    except Error as e:
        return jsonify({"error": str(e)})
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
