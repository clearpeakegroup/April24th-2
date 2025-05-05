import smtplib
from email.message import EmailMessage

def send_report(subject, body, to="admin@example.com"):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = "noreply@finrl.local"
    msg["To"] = to
    with smtplib.SMTP("localhost") as s:
        s.send_message(msg)
