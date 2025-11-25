#!/usr/bin/env python3
"""
Email Sender - Automated email delivery for predictions
Supports multiple delivery methods:
1. SMTP email (Outlook, Gmail, etc.)
2. Future: Telegram bot
3. Future: Discord webhook
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from datetime import datetime
from typing import Optional, List

class EmailSender:
    """
    Send emails with attachments via SMTP

    Configuration via environment variables:
    - EMAIL_SMTP_SERVER: SMTP server (e.g., smtp-mail.outlook.com)
    - EMAIL_SMTP_PORT: SMTP port (e.g., 587)
    - EMAIL_SENDER: Your email address
    - EMAIL_PASSWORD: Your email password or app password
    - EMAIL_RECIPIENT: Recipient email address
    """

    def __init__(self):
        self.smtp_server = os.environ.get('EMAIL_SMTP_SERVER', 'smtp-mail.outlook.com')
        self.smtp_port = int(os.environ.get('EMAIL_SMTP_PORT', '587'))
        self.sender = os.environ.get('EMAIL_SENDER', '')
        self.password = os.environ.get('EMAIL_PASSWORD', '')
        self.recipient = os.environ.get('EMAIL_RECIPIENT', '')

    def is_configured(self) -> bool:
        """Check if email is properly configured"""
        return bool(self.sender and self.password and self.recipient)

    def send_predictions(self,
                        excel_file: Path,
                        subject: Optional[str] = None,
                        body: Optional[str] = None,
                        additional_files: Optional[List[Path]] = None) -> bool:
        """
        Send predictions Excel file via email

        Args:
            excel_file: Path to Excel file with predictions
            subject: Email subject (auto-generated if None)
            body: Email body (auto-generated if None)
            additional_files: Optional list of additional files to attach

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_configured():
            print(" Email not configured")
            print("   Set these environment variables:")
            print("   - EMAIL_SMTP_SERVER")
            print("   - EMAIL_SMTP_PORT")
            print("   - EMAIL_SENDER")
            print("   - EMAIL_PASSWORD")
            print("   - EMAIL_RECIPIENT")
            return False

        try:
            # Generate default subject/body if not provided
            if subject is None:
                date_str = datetime.now().strftime('%A, %d %B %Y')
                subject = f" Football Predictions - {date_str}"

            if body is None:
                body = self._generate_email_body(excel_file)

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = self.recipient
            msg['Subject'] = subject

            # Attach body
            msg.attach(MIMEText(body, 'html'))

            # Attach main Excel file
            self._attach_file(msg, excel_file)

            # Attach additional files if provided
            if additional_files:
                for file_path in additional_files:
                    if file_path.exists():
                        self._attach_file(msg, file_path)

            # Send email
            print(f"\n Sending email to {self.recipient}...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.send_message(msg)

            print(f" Email sent successfully!")
            print(f"    Attached: {excel_file.name}")
            if additional_files:
                for f in additional_files:
                    print(f"    Attached: {f.name}")

            return True

        except Exception as e:
            print(f" Failed to send email: {e}")
            return False

    def _attach_file(self, msg: MIMEMultipart, file_path: Path):
        """Attach a file to the email message"""
        with open(file_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())

        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {file_path.name}'
        )
        msg.attach(part)

    def _generate_email_body(self, excel_file: Path) -> str:
        """Generate HTML email body"""
        date_str = datetime.now().strftime('%A, %d %B %Y')

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: #f8f9fa;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                }}
                .highlight {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 4px solid #667eea;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #666;
                    font-size: 12px;
                }}
                ul {{
                    list-style-type: none;
                    padding: 0;
                }}
                li {{
                    padding: 8px 0;
                }}
                li:before {{
                    content: " ";
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1> Football Predictions</h1>
                    <p>{date_str}</p>
                </div>
                <div class="content">
                    <p>Hi there! </p>

                    <p>Your weekly football predictions are ready!</p>

                    <div class="highlight">
                        <h3> What's Included:</h3>
                        <ul>
                            <li><strong>Top 10 All Markets</strong> - Best picks across all markets</li>
                            <li><strong>BTTS Predictions</strong> - Both Teams To Score analysis</li>
                            <li><strong>Over/Under Markets</strong> - Goal line predictions (0.5 to 5.5)</li>
                        </ul>
                    </div>

                    <div class="highlight">
                        <h3> System Features:</h3>
                        <ul>
                            <li>Dixon-Coles statistical model</li>
                            <li>Rest days analysis (fixture congestion)</li>
                            <li>Seasonal goal patterns</li>
                            <li>Calibrated probabilities</li>
                        </ul>
                    </div>

                    <p><strong> Attachment:</strong> {excel_file.name}</p>

                    <p>Good luck! </p>

                    <div class="footer">
                        <p>Automated prediction system powered by Dixon-Coles</p>
                        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html


def send_weekly_predictions(excel_file: Path, html_report: Optional[Path] = None) -> bool:
    """
    Convenience function to send weekly predictions

    Args:
        excel_file: Path to Excel predictions file
        html_report: Optional HTML report file to attach

    Returns:
        True if sent successfully
    """
    sender = EmailSender()

    if not sender.is_configured():
        print("\n TIP: Configure email sending by setting environment variables")
        print("   in run_weekly.py or your shell profile (.bashrc, .zshrc)")
        return False

    additional_files = [html_report] if html_report and html_report.exists() else None

    return sender.send_predictions(
        excel_file=excel_file,
        additional_files=additional_files
    )


if __name__ == "__main__":
    # Test email configuration
    sender = EmailSender()

    if sender.is_configured():
        print(" Email is configured")
        print(f"   Server: {sender.smtp_server}:{sender.smtp_port}")
        print(f"   Sender: {sender.sender}")
        print(f"   Recipient: {sender.recipient}")
    else:
        print(" Email is not configured")
        print("\nSet these environment variables:")
        print("   export EMAIL_SMTP_SERVER='smtp-mail.outlook.com'")
        print("   export EMAIL_SMTP_PORT='587'")
        print("   export EMAIL_SENDER='your-email@outlook.com'")
        print("   export EMAIL_PASSWORD='your-password'")
        print("   export EMAIL_RECIPIENT='recipient@email.com'")
