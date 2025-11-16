# email_sender.py
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime

def send_predictions_email(html_file_path: Path, csv_file_path: Path = None):
    """Send the predictions HTML file via email"""
    
    # Email configuration from environment variables
    # Email configuration from environment variables
smtp_server = os.environ.get("EMAIL_SMTP_SERVER", "smtp-mail.outlook.com")
smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
    sender_email = os.environ.get("EMAIL_SENDER")
    sender_password = os.environ.get("EMAIL_PASSWORD")
    recipient_email = os.environ.get("EMAIL_RECIPIENT")
    
    if not all([sender_email, sender_password, recipient_email]):
        print("❌ Email configuration missing. Set EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT")
        return False
    
    try:
        # Read the HTML file
        html_content = html_file_path.read_text(encoding='utf-8')
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Football Predictions - {datetime.now().strftime('%Y-%m-%d')}"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Optionally attach CSV file
        if csv_file_path and csv_file_path.exists():
            from email.mime.base import MIMEBase
            from email import encoders
            
            with open(csv_file_path, 'rb') as f:
                csv_part = MIMEBase('application', 'octet-stream')
                csv_part.set_payload(f.read())
            encoders.encode_base64(csv_part)
            csv_part.add_header(
                'Content-Disposition',
                f'attachment; filename= {csv_file_path.name}'
            )
            msg.attach(csv_part)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

if __name__ == "__main__":
    # Test the email sender
    html_path = Path("outputs/top50.html")
    csv_path = Path("outputs/weekly_bets.csv")
    send_predictions_email(html_path, csv_path)