# ===============================================================================
""" Все опции для запуска берутся из конфига """
Thread = False

prototxt = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
input = 'videos/example_01.mp4'
# input = 'http://158.58.130.148/mjpg/video.mjpg'  # люди
#input = 'http://83.56.31.69/mjpg/video.mjpg'
output = None
isstream = False
skip_frames = 30  # скип кадра, если в нем нет людей
confidence = 0.3
log = True
# ===============================================================================
