from flask import Flask, render_template, request, redirect, url_for
from flask_mail import Mail, Message
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '****'# add ur gmail
app.config['MAIL_PASSWORD'] = '****'# add ur App password
app.config['MAIL_DEFAULT_SENDER'] = '****'#add ur gmail
mail = Mail(app)

# Load the trained model
model = joblib.load("best_svm_model.pkl")

# List to store user grades
user_grades = []

@app.route("/")
def main_page():
    return render_template("main.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        try:
            # Collect grades, email, and additional information from form
            english_grade = float(request.form["english_grade"])
            math_grade = float(request.form["math_grade"])
            sciences_grade = float(request.form["sciences_grade"])
            language_grade = float(request.form["language_grade"])
            nationality = request.form["nationality"]
            city = request.form["city"]
            gender = request.form["gender"]
            ethnic_group = request.form["ethnic_group"]
            email = request.form["email"]

            # Prepare input for model prediction
            features = pd.DataFrame({
                'english.grade': [english_grade],
                'math.grade': [math_grade],
                'sciences.grade': [sciences_grade],
                'language.grade': [language_grade],
                'nationality': [nationality],
                'city': [city],
                'gender': [gender],
                'ethnic.group': [ethnic_group]
            })
            prediction = model.predict(features)[0]
            performance = "Top Student" if prediction == 1 else "Low Student"

            # Store the current grades
            user_grades.append({
                "english": english_grade,
                "math": math_grade,
                "sciences": sciences_grade,
                "language": language_grade,
                "performance": performance,
                "nationality": nationality,
                "city": city,
                "gender": gender,
                "ethnic_group": ethnic_group
            })

            # Send email with prediction results
            subject = "Your Performance Prediction Results"
            body = f"""
            Hello,

            Here are your performance results:
            - English Grade: {english_grade}
            - Math Grade: {math_grade}
            - Sciences Grade: {sciences_grade}
            - Language Grade: {language_grade}
            - Performance: {performance}

            Best regards,
            Grade Guard Team
            """
            msg = Message(subject, recipients=[email])
            msg.body = body
            mail.send(msg)

            return redirect(url_for("report"))
        except ValueError:
            return render_template("index.html", error="Please enter valid grades.")
        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {str(e)}")

    return render_template("index.html")

@app.route("/report")
def report():
    if not user_grades:
        return "No grades uploaded yet."

    # Calculate average grades
    average_grades = {
        "english": sum(g["english"] for g in user_grades) / len(user_grades),
        "math": sum(g["math"] for g in user_grades) / len(user_grades),
        "sciences": sum(g["sciences"] for g in user_grades) / len(user_grades),
        "language": sum(g["language"] for g in user_grades) / len(user_grades),
    }

    # Show latest grade and overall averages
    latest_grades = user_grades[-1]
    return render_template("report.html", latest_grades=latest_grades, average_grades=average_grades)

if __name__ == "__main__":
    app.run(debug=True)
