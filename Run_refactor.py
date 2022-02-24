from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
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

    def __log_to_csv(self, empty1, empty, x) -> None:
        """Логирование результатов в csv файл"""
        # TODO: перепиши эту муть а...
        # d = [empty1, empty, x]
        # export_data = zip_longest(*d, fillvalue='')

        # with open('Log.csv', 'w', newline='') as myfile:
        #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #     wr.writerow(("In", "Out", "Total Inside"))

        #     wr.writerows(export_data)
        pass

    def __draw_window_output(self, frame, info, info2, H) -> bool:
        # Вывод информации
        for (i, (k, v)) in enumerate(info):
            text = f'{k}: {v}'
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

    def __draw_visual_line(self, frame, W, H, i) -> None:
        """Зарисовка визуальной линии-прохода на кадре"""
        # h // 2 было
        cv2.line(frame, (0, (H // 2)), (W, (H // 2)), (0, 0, 0), 3)
        cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def __draw_object(self, frame,  centroid, objectID) -> None:
        """Зарисовка ID объекта и зарисовка его на кадре"""
        text = f'ID {objectID}'
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        pass

    def __create_detection_line(self):
        pass

    def __grab_stream(self):
        """Захват видеопотока"""
        if self.isstream:
            print("[INFO] Starting the live stream..")
            self.vs = VideoStream(self.input).start()
            time.sleep(1.0)
        else:
            print("[INFO] Starting the video..")
            self.vs = cv2.VideoCapture(self.input)

    def __final_console_output(self, fps):
        """Вывод информации в консоль по завершению"""
        fps.stop()
        print(f"[INFO] elapsed time: {fps.elapsed():.2f}")
        print(f"[INFO] approx. FPS: {fps.fps():.2f}")

    def __detect():
        pass

    def __init_writer(self):
        pass

    def run(self):
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

        self.__grab_stream()

        # Перенести CT в инит?
        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.writer = None
        self.trackers = []
        self.trackableObjects = {}
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0
        self.people_inside = []
        self.total_up_list = []
        self.total_down_list = []
        self.fps = FPS().start()
        VIDEO_WIDTH = None
        VIDEO_HEIGHT = None

        while True:
            frame = self.vs.read()
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
            if self.output is not None and self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(
                    self.output, fourcc, 30, (VIDEO_WIDTH, VIDEO_HEIGHT), True)

            # Инициализация статуса (визуально) и
            # 	списка для описывающих коробок, который получаем
            # 	от детектера или трекера
            status = "Waiting"
            rects = []

            # Object-detection в помощь Object-tracker'у,
            # проверка на необходить детекта
            if self.totalFrames % self.skip_frames == 0:
                status = "Detecting"
                self.trackers = []

                # конвертация кадра в блоб и прогон через нейронку
                blob = cv2.dnn.blobFromImage(
                    frame, 0.007843, (VIDEO_WIDTH, VIDEO_HEIGHT), 150)
                net.setInput(blob)
                detections = net.forward()

                # цикл по детектам
                for i in np.arange(0, detections.shape[2]):
                    # получение уверенности
                    confidence_gotten = detections[0, 0, i, 2]
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue
                    if not confidence_gotten > self.confidence:
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
                    self.trackers.append(tracker)

            # Если нужное кол-во кадрво не прошло, занимаемся
            # 	трекингом, вместо детектинга
            else:
                for tracker in self.trackers:
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

            self.__draw_visual_line(frame, VIDEO_WIDTH, VIDEO_HEIGHT, i)

            # Использованеи центроид трекера для связки старых обьектов-центроидов
            # 	(чтобы не создавать новые) и новых границ обьектов
            objects = ct.update(rects)
            for (objectID, centroid) in objects.items():

                # Проверка существует ли отслеживаемый обьект для выдранного ID
                to = self.trackableObjects.get(objectID, None)

                # Если отсуствует - создаем
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # Если существует - используем существующий
                else:
                    # Разница между y-координатой *настоящего* центроида
                    # и ср. арифм. предыдущего позволяет нам понать двигается обьект
                    # вверх (отрицательное) или вниз (положительное)

                    # для более сложных линий придется поменять, наверное
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    # TODO: Сделать нормальный метод подсчета
                    if not to.counted:
                        # Если обьект двигается вверх И центроид выше линии - подсчитываем
                        # Тут VIDEO_HEIGHT // 2 - ровно центральная линия
                        if direction < 0 and centroid[1] < VIDEO_HEIGHT // 2:
                            self.totalUp += 1
                            self.total_up_list.append(self.totalUp)
                            to.counted = True

                        # Если вниз
                        elif direction > 0 and centroid[1] > VIDEO_HEIGHT // 2:
                            self.totalDown += 1
                            self.total_down_list.append(self.totalDown)
                            to.counted = True

                        self.people_inside = []
                        # Подсчет суммы людей внутри
                        self.people_inside.append(
                            len(self.total_down_list)-len(self.total_up_list))

                self.trackableObjects[objectID] = to
                self.__draw_object(frame, centroid, objectID)

            # Подготовка информации для вывода
            info1 = [
                ("WentUp", self.totalUp),
                ("WentDown", self.totalDown),
                ("Status", status),
            ]
            info2 = [("Total people inside", self.people_inside), ]

            if self.log:
                self.__log_to_csv(
                    self.total_up_list, self.total_down_list, self.people_inside)

            # Запись получившегося кадра
            if self.writer is not None:
                self.writer.write(frame)

            if self.__draw_window_output(frame, info1, info2, VIDEO_HEIGHT):
                break

            self.totalFrames += 1
            self.fps.update()

        self.__final_console_output(self.fps)

        if config.Thread:
            self.vs.release()

        cv2.destroyAllWindows()


def main():
    ay = Detection(input=cfg.input)
    ay.run()


if __name__ == "__main__":
    main()
