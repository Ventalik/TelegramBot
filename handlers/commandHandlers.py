from aiogram import types
from aiogram.dispatcher import filters
from main import dp, bot
from states import BotStates
from messages import MESSAGES
from keyboards import *


@dp.message_handler(filters.Text(equals=START_BTN_NAMES), state='*')
async def process_start_keyboard(msg: types.Message):
    """
    Обрабатывает нажатия на стартовой клавиатуре
    и меняет состояние бота
    """

    state = dp.current_state(user=msg.from_user.id)
    message = ''
    kb = None
    if msg.text == 'Передача стиля':
        await state.set_state(BotStates.STYLE_TRANSFER_STATE_1)
        message = MESSAGES['style transfer']
    elif msg.text == 'Нанесение стиля':
        await state.set_state(BotStates.STYLE_GAN_STATE)
        message = MESSAGES['style GAN']
        kb = style_kb
    elif msg.text == 'Повышение качества':
        await state.set_state(BotStates.SUPER_RESOLUTION_STATE)
        message = MESSAGES['super resolution']
    await bot.send_message(msg.from_user.id, message, reply_markup=kb)


@dp.message_handler(filters.Text(equals=STYLE_BTN_NAMES), state='*')
async def process_style_keyboard(msg: types.Message):
    """
    Обрабатывает нажатия на клавиатуре выбора стиля
    и меняет состояние бота
    """

    state = dp.current_state(user=msg.from_user.id)

    if msg.text == 'Кубизм':
        await state.set_state(BotStates.CUBISM_STATE)
    elif msg.text == 'Ранний Ренессанс':
        await state.set_state(BotStates.RENAISSANCE_STATE)
    elif msg.text == 'Экспресионизм':
        await state.set_state(BotStates.EXPRESSIONISM_STATE)

    await bot.send_message(msg.from_user.id, MESSAGES['request photo'], reply_markup=start_kb)


@dp.message_handler(filters.Text(equals='Назад'), state='*')
async def process_back_keyboard(msg: types.Message):
    """
    Обрабатывает нажатие кнопки 'Назад'
    и возвращаает пользователя в начальное меню и состояние
    """

    state = dp.current_state(user=msg.from_user.id)
    await state.set_state(BotStates.START_STATE)
    await bot.send_message(msg.from_user.id, MESSAGES['selecting action'], reply_markup=start_kb)


@dp.message_handler(state='*', commands=['start', 'help'])
async def process_start_command(msg: types.Message):
    """
    Обрабатывает команды старта и помощи.
    Выводит пользователю подсказки о работе с ботом
    """

    state = dp.current_state(user=msg.from_user.id)
    await state.set_state(BotStates.START_STATE)
    await bot.send_message(msg.from_user.id, MESSAGES['hello'], reply_markup=start_kb)
    await bot.send_message(msg.from_user.id, MESSAGES['help1'])
    await bot.send_message(msg.from_user.id, MESSAGES['help2'])
    await bot.send_message(msg.from_user.id, MESSAGES['help3'])
    await bot.send_message(msg.from_user.id, MESSAGES['help4'])


@dp.message_handler(state='*')
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)
