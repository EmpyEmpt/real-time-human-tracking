from dataclasses import dataclass
import urllib.request
import re
import ssl


@dataclass
class TrassirCredentials:
    link: str                       # Ссылка на трассив (192.168.0.1 например)
    sid: str = None                 # Токен сессии, получаем после входа
    sdk_password: str = None        # Пароль от SDK
    login: str = None               # Логин произвольного пользователя
    user_password: str = None       # Пароль произвольного пользователя


class TrassirAuth:
    """Аутентификация в Трассире"""

    context = ssl._create_unverified_context()

    def auth_via_SDK(self, credentials: TrassirCredentials) -> TrassirCredentials:
        """Аутентификация с помощью пароля SDK
        Возвращает TrassirCredentials с SIDом"""
        s = urllib.request.urlopen('https://' + credentials.link + ':8080/login?password=' +
                                   credentials.sdk_password,
                                   context=self.context).read().decode('utf-8')

        sid = re.search('"sid"\s*:\s*"(.*?)"', s).group(1)
        credentials.sid = sid
        return credentials

    def auth_via_password(self, credentials: TrassirCredentials) -> TrassirCredentials:
        """Аутентификация с помощью логина и пароля пользователя
        Возвращает TrassirCredentials с SIDом"""

        s = urllib.request.urlopen('https://' + credentials.link + ':8080/login?username=' +
                                   credentials.login + '&password=' +
                                   credentials.user_password, context=self.context).read().decode('utf8')

        sid = re.search('"sid"\s*:\s*"(.*?)"', s).group(1)
        credentials.sid = sid
        return credentials
