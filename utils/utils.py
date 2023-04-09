from sys import stdout
from time import time, sleep
from utils.logger import ProjectLogger
from utils.threading import Consumer

import wave
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_ctx(args):
    devices_id = [int(i) for i in args.gpus.split(',') if i.strip()]
    if torch.cuda.is_available():
        if len(devices_id) == 0:
            devices_id = list(range(torch.cuda.device_count()))

        ctx = [torch.device(f'cuda:{i}') for i in devices_id if i >= 0]
        ctx = ctx if len(ctx) > 0 else [torch.device('cpu')]
    else:
        ProjectLogger().error('Cannot access GPU.')
        ctx = [torch.device('cpu')]

    ProjectLogger().info('Used context: {}'.format(', '.join([str(x) for x in ctx])))
    return ctx


def save_to_file(path, data, sample_width, sampling_rate=16000):
    """Records from the microphone and outputs the resulting data to 'path'."""
    # sample_width, data = record()
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sampling_rate)
        wf.writeframes(data)


class TypeWriter(Consumer):

    @staticmethod
    def sanitize(transcription):
        transcription = list(transcription)
        for i in range(len(transcription)):
            transcription[i] = transcription[i].lower() if i > 0 else transcription[i].upper()
            in_sentence_with_letters_around = 0 < i < len(transcription) - 1 and transcription[i - 1] != '' and transcription[i + 1] != ''
            one_letter_word = i - 2 < 0 or transcription[i - 2] == ' '
            is_space = transcription[i] == ' '
            valid_letter = transcription[i - 1].lower() in 'jmstcndl'
            if is_space and in_sentence_with_letters_around and one_letter_word and valid_letter:
                transcription[i] = '\''
        return ''.join(transcription)

    @staticmethod
    def display(transcription):
        if len(transcription) > 0:
            stdout.write('User : ')
            stdout.flush()
            for char in transcription:
                stdout.write(char)
                stdout.flush()
                sleep(.05)
            print('.')

    def run(self) -> None:
        while True:
            text = self._in_queue.get()
            TypeWriter.display(text)


class LivePlotter:
    def __init__(self, title, chunk_duration, chunk_size, bits=16):
        self._chunk_size = chunk_size

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12.8, 7.2))

        self.x = np.linspace(0, chunk_duration, num=chunk_size)

        self.line1 = self.ax.plot([], lw=1)[0]
        self.line1.set_color('tab:blue')

        self.line2 = self.ax.plot([], lw=1)[0]
        self.line2.set_color('tab:red')

        self.line3 = self.ax.plot([], lw=1)[0]
        self.line3.set_color('tab:blue')
        # self.text = self.ax.text(0.8, 0.5, '')

        self.ax.set_title(title)
        self.ax.set_xlim(self.x.min(), self.x.max())
        # lim = (2 ** bits - 1) // 2
        lim = 1
        self.ax.set_ylim([-lim, lim])

        # note that the first draw comes before setting data
        self.fig.canvas.draw()
        # cache the background
        self.ax_bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        plt.show(block=False)

        self.t_start = time()

    def draw(self, y, speech_start, speech_end):
        if type(y) != np.ndarray:
            y = y.numpy()

        y = np.pad(y, (0, max(self._chunk_size - len(y), 0)))

        # fps_count = ((index + 1) / (time() - self.t_start))
        # tx = f'Mean Frame Rate:\n {fps_count:.3f}FPS'
        # self.text.set_text(tx)
        if speech_start is None:
            self.line1.set_data(self.x, y)
            self.line2.set_data([], [])
            self.line3.set_data([], [])
        else:
            speech_start = int(speech_start)
            speech_end = int(speech_end)
            x1, y1 = self.x[:speech_start], y[:speech_start]
            x2, y2 = self.x[speech_start:speech_end], y[speech_start:speech_end]
            x3, y3 = self.x[speech_end:], y[speech_end:]

            if len(x1) > 0:
                self.line1.set_data(x1, y1)
            else:
                self.line1.set_data([], [])

            if len(x2) > 0:
                self.line2.set_data(x2, y2)
            else:
                self.line2.set_data([], [])

            if len(x3) > 0:
                self.line3.set_data(x3, y3)
            else:
                self.line3.set_data([], [])

        # restore background
        self.fig.canvas.restore_region(self.ax_bg)

        # redraw just the points
        # ax1.draw_artist(img)
        self.ax.draw_artist(self.line1)
        self.ax.draw_artist(self.line2)
        self.ax.draw_artist(self.line3)
        # self.ax.draw_artist(self.text)

        # fill in the axes rectangle
        self.fig.canvas.blit(self.ax.bbox)

        # in this post http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
        # it is mentionned that blit causes strong memory leakage.
        # however, I did not observe that.

        self.fig.canvas.flush_events()


def frame_decode(frame):
    decoded = dict()
    frame_copy = frame.copy()
    while True:
        chunk_header = frame_copy[:3].decode('utf-8')
        chunk_size = int.from_bytes(frame_copy[3:7], 'big')
        chunk_content = frame_copy[7:7+chunk_size]
        if len(chunk_content) < chunk_size:
            return None

        if chunk_header == 'PCM':
            decoded[chunk_header] = chunk_content
        elif chunk_header == 'ANS':
            decoded['IDX'] = chunk_content[0]
            decoded[chunk_header] = chunk_content[1:].decode('utf-8')
        else:
            decoded[chunk_header] = chunk_content.decode('utf-8')

        frame_copy = frame_copy[7+chunk_size:]
        if chunk_header == 'PCM':
            return decoded, frame_copy


def frame_encode(idx, request, answer, pcm):
    # beware of accents, they are using 2 bytes. Byte string might be longer than str
    answer = int.to_bytes(idx, 1, 'big') + bytes(answer, 'utf-8')
    request = bytes(request, 'utf-8')

    req_len = len(request)
    ans_len = len(answer)
    pcm_len = len(pcm) * 2  # because each value is coded on 2 bytes (16 bits)

    # print(f'req {req_len}')
    # print(f'ans {ans_len}')
    # print(f'pcm {pcm_len}')

    frame = bytes('REQ', 'utf-8')
    frame += req_len.to_bytes(4, 'big')  # big = read bytes from left to right
    frame += request

    frame += bytes('ANS', 'utf-8')
    frame += ans_len.to_bytes(4, 'big')
    frame += answer

    frame += bytes('PCM', 'utf-8')
    frame += pcm_len.to_bytes(4, 'big')
    frame += pcm.tobytes()

    # print(f'Frame len {len(frame)}')
    return frame
