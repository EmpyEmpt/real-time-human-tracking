from dataclasses import dataclass
import urllib.request
import re
import ssl
from trassirauth import TrassirCredentials
from trassirchannels import TrassirChannel
from PIL import Image


@dataclass
class VideoStreamReqParameters:
    channel: TrassirChannel
    container: str                  # тип контейнера
    quality: str                    # качество 1-100
    stream: str                     # main - sub - archive_main - archive_sub
    framerate: str                  # 1000 - 1 кадр в секунду, 60000 макс принимаемое значение


class VideoRequest:
    """Класс запроса видео потока"""
    parameters: VideoStreamReqParameters
    token: str
    context = ssl._create_unverified_context()

    def __init__(self, parameters: VideoStreamReqParameters):
        self.parameters = parameters

    def request_token(self, credentials: TrassirCredentials) -> str:
        """Получение токена потока"""
        s = urllib.request.urlopen('https://' + credentials.link + ':8080/get_video?' +
                                   'channel=' + self.parameters.channel.guid +
                                   '&container=' + self.parameters.container +
                                   '&quality=' + self.parameters.quality +
                                   '&stream=' + self.parameters.stream +
                                   '&framerate=' + self.parameters.framerate +
                                   '&sid=' + credentials.sid, context=self.context).read().decode('utf-8')
        
        #print('https://' + credentials.link + ':8080/get_video?' +
                                   #'channel=' + self.parameters.channel.guid +
                                   #'&container=' + self.parameters.container +
                                   #'&quality=' + self.parameters.quality +
                                   #'&stream=' + self.parameters.stream +
                                   #'&framerate=' + self.parameters.framerate +
                                   #'&sid=' + credentials.sid)
        print(s)
        for x in s.replace("\n", "").split("{"):
            m = re.search('"token"\s*\:\s*"(.*?)"', x)

            if not m:
                continue
            self.token = m.group(1)
            return self.token

    # Дефолтный порт запросов к видео-потоку - 555
    def _get_stream(self, credentials: TrassirCredentials) -> str:
        """Получение видеопотока -> ссылка на поток"""
        # s = urllib.request.urlopen('https://' + credentials.link + ':555/' + self.token,
        #                            context=self.context)
        s = 'https://' + credentials.link + ':555/' + self.token
        return s

    def keep_alive(self):
        """Поддержание жизни токена (стандартное время жизни - 10 секунд)"""
        urllib.request.urlopen('https://' + self.params.credentials.link + ':555/' + self.token + '?ping',
                               context=self.context)
