from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

START_BTN_NAMES = ['Передача стиля', 'Нанесение стиля', 'Повышение качества']
start_btns = [KeyboardButton(name) for name in START_BTN_NAMES]

start_kb = ReplyKeyboardMarkup(resize_keyboard=True)
start_kb.row(*start_btns)


STYLE_BTN_NAMES = ['Кубизм', 'Ранний Ренессанс', 'Экспресионизм']
start_btns = [KeyboardButton(name) for name in STYLE_BTN_NAMES]
back_btn = KeyboardButton('Назад')

style_kb = ReplyKeyboardMarkup(resize_keyboard=True)
style_kb.row(*start_btns)
style_kb.row(back_btn)
