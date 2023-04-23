import tkinter as tk


class AudioLevel(tk.Canvas):
    def __init__(self, master=None, width=0, height=0, num_bar=15):
        super().__init__(master, width=width, height=height, bg='#242424', highlightthickness=0)
        self.level = 0
        self.bars = []
        self.padding_size = 6
        self.bar_width = (width - ((num_bar - 1) * self.padding_size)) // num_bar
        used_size = (num_bar - 1) * self.padding_size + self.bar_width * num_bar
        margin = (width - used_size) // 2
        for i in range(15):
            padding = 0 if i == 0 else self.padding_size
            start_pos = i * padding + i * self.bar_width + margin
            top_left = (start_pos, height)
            bottom_right = (start_pos + self.bar_width, 0)
            bar = self.create_rectangle(*top_left, *bottom_right, fill='#c2c2c2', outline='')
            # bar = self.round_rectangle(*top_left, *bottom_right, r=50, fill='#c2c2c2', outline='')
            self.bars.append(bar)

    def set_level(self, level):
        if level != self.level:
            for i in range(15):
                if i < level:
                    self.itemconfigure(self.bars[i], fill='white')
                else:
                    self.itemconfigure(self.bars[i], fill='grey')

            self.level = level

    def round_rectangle(self, x1, y1, x2, y2, r=50, **kwargs):
        points = (
            x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y1 + r, x2, y2 - r, x2, y2 - r, x2, y2,
            x2 - r, y2, x2 - r, y2, x1 + r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y2 - r, x1, y1 + r, x1, y1 + r, x1, y1
        )
        return self.create_polygon(points, **kwargs, smooth=True)
