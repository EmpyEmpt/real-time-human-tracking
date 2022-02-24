from trassirauth import TrassirCredentials, TrassirAuth
from trassirchannels import TrassirChannel
from trassirrequest import VideoStreamReqParameters, VideoRequest
from Run_refactor import Detection

"""Попытки тестировать трассир"""


def main():
    credentials = TrassirCredentials(
        link='10.2.120.235',
        login='operator',
        user_password='gdevideo')

    auth = TrassirAuth()
    credentials = auth.auth_via_password(credentials)

    channel = TrassirChannel()
    channel._print_channel_tree(credentials=credentials)
    channel.guid = 'qVOOTR7T'
    # channel._get_guid_by_name(
    # name='Наб. Свияги, 106, 1 корпус - крыльцо', credentials=credentials)

    parameters = VideoStreamReqParameters(
        channel=channel,
        container='mjpeg',
        quality='100',
        stream='main',
        framerate='30000')

    stream = VideoRequest(parameters=parameters)
    stream.request_token(credentials=credentials)
    link = stream._get_stream(credentials=credentials)
    prototxt = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
    model = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
    # input = 'videos/example_01.mp4'
    #input = link
    output = None
    skip_frames = 1  # скип кадра, если в нем нет людей
    confidence = 0
    print(link)
    ay = Detection(input=link)
    ay.run()

    # https://10.2.120.235:8080/login?username=operator&password=gdevideo
    # https://10.2.120.235:8080/get_video?channel=x0uRf8dD&container=mjpeg&quality=50&stream=main&framerate=10000&sid=ZqIM2JFW
    # https://10.2.120.235:555/uqgRFeCk


if __name__ == '__main__':
    main()
