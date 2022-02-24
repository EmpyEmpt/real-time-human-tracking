import json
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
# from mylib.mailer import Mailer
from mylib import config, thread
from itertools import zip_longest
import time
import csv
import numpy as np
import imutils
import time
import dlib
import cv2
import datetime
import mylib.config as cfg
import requests
import json
import cv2

t0 = time.time()


class Detection:
    """God's view for recognizing people"""
    prototxt: str
    model: str
    input: str = None
    output: str = None
    isstream: bool
    skip_frames: int = 30
    confidence: float = 0
    log: bool

    def __init__(self, input):
        self.prototxt = cfg.prototxt
        self.model = cfg.model
        # if cfg.input:
        #self.input = cfg.input
        self.input = input
        if cfg.output:
            self.output = cfg.output
        self.isstream = cfg.isstream
        self.skip_frames = cfg.skip_frames
        self.confidence = cfg.confidence
        self.log = cfg.log

    def __log_to_csv__(self, empty1, empty, x) -> None:
        """Логирование результатов в csv файл"""
        datetimee = [datetime.datetime.now()]
        d = [datetimee, empty1, empty, x]
        export_data = zip_longest(*d, fillvalue='')

        with open('Log.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(("End Time", "In", "Out", "Total Inside"))
            wr.writerows(export_data)

    def __draw_window_output__(self, frame, info, info2, H) -> bool:
        # Вывод информации
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(
                frame, text,
                (175, H - ((i * 20) + 60)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

        cv2.imshow("Aboba", frame)
        key = cv2.waitKey(1) & 0xFF
        # Выход по нажатию q
        if key == ord("q"):
            return True
        return False

    def __draw_visual_line__(self, frame, W, H, i) -> None:
        """Зарисовка визуальной линии-прохода на кадре"""
        # h // 2 было
        cv2.line(frame, (0, (H // 2)), (W, (H // 2)), (0, 0, 0), 3)
        cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def __draw_object__(self, frame,  centroid, objectID) -> None:
        """Зарисовка ID объекта и зарисовка его на кадре"""
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        pass

    def __create_detection_line__(self):
        pass

    def __grab_stream__(self):
        """Захват видеопотока"""
        if self.isstream:
            print("[INFO] Starting the live stream..")
            self.vs = VideoStream(self.input).start()
            time.sleep(1.0)
        else:
            print("[INFO] Starting the video..")
            self.vs = cv2.VideoCapture(self.input)

    def __final_console_output__(self, fps):
        """Вывод информации в консоль по завершению"""
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    def __init_writer__(self):
        pass

    def run(self):
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

        self.__grab_stream__()

        # Перенести CT в инит?
        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        writer = None
        trackers = []
        trackableObjects = {}
        totalFrames = 0
        totalDown = 0
        totalUp = 0
        people_inside = []
        total_up_list = []
        total_down_list = []
        fps = FPS().start()
        VIDEO_WIDTH = None
        VIDEO_HEIGHT = None

        while True:
            frame = self.vs.read()
            print(type(frame))
            # Для видео
            if not self.isstream:
                frame = frame[1]

            if self.input is not None and frame is None:
                break
            # Выбор ширины и высоты кадра по значениям кадра, если изначално не заданы
            if VIDEO_WIDTH is None or VIDEO_HEIGHT is None:
                (VIDEO_HEIGHT, VIDEO_WIDTH) = frame.shape[:2]
            # Сжатие видео до 500 кадров в шириру, перевод из BGR в RGB для dlib
            frame = imutils.resize(frame, width=VIDEO_WIDTH)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Запись видео-результата, если указано
            # Перенести в __init_writer__ ?
            if self.output is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    self.output, fourcc, 30, (VIDEO_WIDTH, VIDEO_HEIGHT), True)

            # Инициализация статуса (визуально) и
            # 	списка для описывающих коробок, который получаем
            # 	от детектера или трекера
            status = "Waiting"
            rects = []

            # Object-detection в помощь Object-tracker'у,
            # проверка на необходить детекта
            if totalFrames % self.skip_frames == 0:
                # Перенести в self.detect ?
                status = "Detecting"
                trackers = []

                # конвертация кадра в блоб и прогон через нейронку
                blob = cv2.dnn.blobFromImage(
                    frame, 0.007843, (VIDEO_WIDTH, VIDEO_HEIGHT), 250)
                net.setInput(blob)
                detections = net.forward()

                # цикл по детектам
                for i in np.arange(0, detections.shape[2]):
                    # получение уверенности
                    confidence_gotten = detections[0, 0, i, 2]

                    # Отфильтровка слабых предположений
                    if not confidence_gotten > self.confidence:
                        continue
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    # вычисление x, y координат для описывающей коробки обьекта
                    box = detections[0, 0, i, 3:7] * np.array(
                        [VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_HEIGHT])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Создаем dlib коробку-обьект из координат полученных выше
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # добавляем трекер к списку трекеров
                    trackers.append(tracker)

            # Если нужное кол-во кадрво не прошло, занимаемся
            # 	трекингом, вместо детектинга
            # 	с целью повышения скорости работы
            else:
                # Перенести в self.track ?
                for tracker in trackers:
                    status = "Tracking"

                    # Получаем обновленную позицию
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    # добавляем координаты в список описывающих коробок
                    rects.append((startX, startY, endX, endY))

            self.__draw_visual_line__(frame, VIDEO_WIDTH, VIDEO_HEIGHT, i)

            # Использованеи центроид трекера для связки старых обьектов-центроидов
            # 	(чтобы не создавать новые) и новых границ обьектов
            objects = ct.update(rects)
            for (objectID, centroid) in objects.items():

                # Проверка существует ли отслеживаемый обьект для выдранного ID
                to = trackableObjects.get(objectID, None)
                """Я не совсем понимаю в чем смысл if-else ветвления тут:

				if проверяет привязан ли обьект к центроиду и отслеживается,
				если нет - связывает центроид и ID
				
				но зачем тут else?
				Получается, мы пропускаем один прогон отслеживания 
				(и детекта пересечения линии, соответственно)
				если обьект был только что создан?
				Не лучше ли убрать else и просто продолжить код?
				
				Тем более после ветвления мы в любом случае заносим обьект
				в список
				+ без else небольшой прирост производительности
				Ничего не понимаю..."""
                # Если отсуствует - создаем
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # Если существует - используем существующий
                else:
                    # Разница между y-координатой *настоящего* центроида
                    # и ср. арифм. предыдущего позволяет нам понать двигается обьект
                    # вверх (отрицательное) или вниз (положительное)

                    # для более сложных линий (как у нас) придется поменять, наверное
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    # Проверка был ли обьект подсчитан
                    if not to.counted:
                        # Если обьект двигается вверх И центроид выше линии - подсчитываем
                        # Тут VIDEO_HEIGHT // 2 - ровно центральная линия
                        if direction < 0 and centroid[1] < VIDEO_HEIGHT // 2:
                            totalUp += 1
                            total_up_list.append(totalUp)
                            to.counted = True

                        # Если вниз
                        elif direction > 0 and centroid[1] > VIDEO_HEIGHT // 2:
                            totalDown += 1
                            total_down_list.append(totalDown)
                            to.counted = True

                        people_inside = []
                        # Подсчет суммы людей внутри
                        people_inside.append(
                            len(total_down_list)-len(total_up_list))

                # Храним отслеживаемый обьект
                trackableObjects[objectID] = to
                # Зарисовка ID обьекта и центроида на кадре
                self.__draw_object__(frame, centroid, objectID)

            # Подготовка информации для вывода
            # Мне не особо нравится передача информации в два списка
            # 	не лучше в один затолкнуть?
            info1 = [
                ("Exit", totalUp),
                ("Enter", totalDown),
                ("Status", status),
            ]
            info2 = [("Total people inside", people_inside), ]

            if self.log:
                self.__log_to_csv__(
                    total_up_list, total_down_list, people_inside)

            # Запись получившегося кадра
            # В отдельный метод тоже?
            if writer is not None:
                writer.write(frame)

            if self.__draw_window_output__(frame, info1, info2, VIDEO_HEIGHT):
                break

            totalFrames += 1
            fps.update()

        # Остановка таймера и вывод FPS
        self.__final_console_output__(fps)

        if config.Thread:
            self.vs.release()

        cv2.destroyAllWindows()


def main():
    ay = Detection(input=cfg.input)
    ay.run()


if __name__ == "__main__":
    # main()

    CAMERA_URL = "https://5536-188-170-129-90.ngrok.io"
    SID = ""

    r = requests.get(
        f"{CAMERA_URL}/login?username=operator&password=gdevideo").json()
    SID = r['sid']

    r = requests.get(f"{CAMERA_URL}/channels?sid={SID}").text
    channels = json.loads(r.split("/*")[0])['channels']

    r = requests.get(
        f"{CAMERA_URL}/get_video?channel={channels[0]['guid']}&container=hls&segment_duration=1&stream=main&sid={SID}").json()

    VIDEO_URL = f"{CAMERA_URL}/hls/{r['token']}/master.m3u8"
    cam = cv2.VideoCapture(VIDEO_URL)
    cv2.namedWindow("ULSU Cam")
    while True:
        f, im = cam.read()
        imS = cv2.resize(im, (960, 540))
        cv2.imshow("ULSU Cam", imS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
