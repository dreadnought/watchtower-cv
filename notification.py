import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from os.path import basename

import requests
import json

import threading, queue
task_queue = queue.Queue()

def async_worker():
    while True:
        task = task_queue.get()
        #task["logger"].debug("processing %s" % task)
        task["func"](**task["arguments"])
        task_queue.task_done()

threading.Thread(target=async_worker, daemon=True).start()

def queue_notification(func, arguments, logger):
    logger.debug("queuing %s" % func)
    task_queue.put(
        {
            "func": func,
            "arguments": arguments,
            "logger": logger
        }
    )

def send_mail(send_from: str, subject: str, text: str, send_to: list, files=None):
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = ', '.join(send_to)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            ext = f.split('.')[-1:]
            attachedfile = MIMEApplication(fil.read(), _subtype=ext)
            attachedfile.add_header(
                'content-disposition', 'attachment', filename=basename(f))
        msg.attach(attachedfile)

    try:
        smtp = smtplib.SMTP(host="localhost", port=25)
    except:
        print('failed')
        return
    smtp.starttls()
    #smtp.login(username, password)
    sent = True
    try:
        smtp.sendmail(send_from, send_to, msg.as_string())
    except smtplib.SMTPSenderRefused:
        sent = False
    smtp.close()
    return sent

def slack_notification(webhook_url, text):
    slack_data = {'text': text}

    response = requests.post(
        webhook_url, data=json.dumps(slack_data),
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code != 200:
        print('Request to slack returned an error %s, the response is:\n%s' % (response.status_code, response.text))