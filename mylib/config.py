# ===============================================================================
""" Все опции для запуска берутся из конфига """
# ===============================================================================
# Enter mail below to receive real-time email alerts
# e.g., 'email@gmail.com'
# MAIL = ''
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video')
url = ''

# ON/OFF for mail feature. Enter True to turn on the email alert feature.
# ALERT = False
# Set max. people inside limit. Optimise number below: 10, 50, 100, etc.
Threshold = 50
# Threading ON/OFF
Thread = False
# Auto run/Schedule the software to run at your desired time
Scheduler = False
# Auto stop the software after certain a time/hours
Timer = False

prototxt = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
# input = 'videos/example_01.mp4'
input = 'http://158.58.130.148/mjpg/video.mjpg'  # люди
#input = 'http://83.56.31.69/mjpg/video.mjpg'
output = None
isstream = True
skip_frames = 30  # скип кадра, если в нем нет людей
confidence = 0.3
log = True
# ===============================================================================
# ===============================================================================
