from trassirauth import TrassirCredentials
import urllib.request
import re
import ssl


class TrassirChannel:
    """Информация о канале"""
    channel_internal_name: str      # Внутреннее имя канала
    channel_real_name: str          # Имя канала (Корпус-1, например)
    guid: str                       # Внутренняя 'ссылка' на канал
    context = ssl._create_unverified_context()

    # Я тут чтото накосячил
    # def _get_guid_by_name(self, name: str, credentials: TrassirCredentials) -> None:
    #"""Получение guid канала по имени"""
    # s = urllib.request.urlopen('https://' + credentials.link + ':8080/channels?sid=' + credentials.sid,
    # context=self.context)

    # for x in s.replace("\n", "").split("{"):
    #m = re.search('"name"\s*\:\s*"(.*?)".*Channel', x)

    # if not m:
    # continue
    #self.guid = m.group(1)
    #self.channel_internal_name = name

    def _print_channel_tree(self, credentials: TrassirCredentials) -> None:
        """Получение списка каналов. Необходим sid"""

        s = urllib.request.urlopen('https://' + credentials.link
                                   + ':8080/channels?sid=' + credentials.sid,
                                   context=self.context).read().decode('utf-8')
        print(s)

    def set_real_name(self, name: str) -> None:
        """Установка читабельного имени канала"""
        self.channel_real_name = name
