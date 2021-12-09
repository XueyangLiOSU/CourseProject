from tkinter import *
from chat import bot_name, get_response

# setting up color and font parameters
BG_GRAY = "#ffffff"
BG_COLOR = "#7e54c7"
BG_COLOR_2 = "#60dae6"
TEXT_COLOR = "#EAECEE"
TEXT_COLOR_2 = "#4d4d4d"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 16 bold"

class ChatApp:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        message = f"{sender}: {msg}\n\n"
        self.text_widget.configure(cursor='arrow', state=NORMAL)
        self.text_widget.insert(END, message)
        self.text_widget.configure(cursor='arrow', state=DISABLED)

        response = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(cursor='arrow', state=NORMAL)
        self.text_widget.insert(END, response)
        self.text_widget.configure(cursor='arrow', state=DISABLED)

        self.text_widget.see(END)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _setup_main_window(self):
        self.window.title("Chatbot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        # head label
        head_label = Label(self.window, bg=BG_COLOR_2, fg=TEXT_COLOR_2,
                            text="CS410 - Course Project Demo", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # line divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, relheight=0.012, rely=0.07)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relwidth=1, relheight=0.825, rely=0.08)
        self.text_widget.configure(cursor='arrow', state=DISABLED)

        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx = 0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.895)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg=BG_COLOR_2, fg=TEXT_COLOR_2, font=FONT)
        self.msg_entry.place(relwidth=1, relheight=0.03, rely=0.008)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)


if __name__ == "__main__":
    app = ChatApp()
    app.run()