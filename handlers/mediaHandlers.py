from aiogram import types
from main import dp, bot
from PIL import Image
from states import BotStates
from models.inference import get_model, remember_style_image
from messages import MESSAGES
import io


@dp.message_handler(content_types=['photo', 'document'], state=BotStates.STYLE_TRANSFER_STATE_1)
async def process_style_photo(msg: types.Message):
    """
    Сохраняет изображение стиля в промежуточный буфер для передачи стиля.
    Меняет состояние бота на BotStates.STYLE_TRANSFER_STATE_2.
    """

    image = await load_img_from_message(msg)
    await remember_style_image(image)

    state = dp.current_state(user=msg.from_user.id)
    await state.set_state(BotStates.STYLE_TRANSFER_STATE_2)


@dp.message_handler(content_types=['photo', 'document'], state='*')
async def process_photo(msg: types.Message):
    """
    Обрабатывает входящее фото от пользлователя.
    Применяет преобразование к фото в зависимости от текущего состояния бота.
    Отправляет преобразованное изображение пользователю в виде документа.
    """

    state = dp.current_state(user=msg.from_user.id)

    model = await get_model(await state.get_state())
    if model is not None:
        image = await load_img_from_message(msg)
        styled_image = await model.predict(image)
        photo = await load_img_in_buffer(styled_image)

        await bot.send_document(msg.from_user.id, document=photo)
    else:
        await bot.send_message(msg.from_user.id, MESSAGES['unknown photo'])

    await state.set_state(BotStates.START_STATE)


async def load_img_from_message(msg: types.Message):
    """
    Загружает изображение из сообщения
    """

    buffer = io.BytesIO()
    if msg.content_type == 'photo':
        await msg.photo[-1].download(buffer)
    else:
        await msg.document.download(buffer)
    image = Image.open(buffer)

    return image


async def load_img_in_buffer(image):
    """
        Загружает изображение в буфер для дальнейшей отправки
    """

    buffer = io.BytesIO()
    buffer.name = 'output.jpeg'

    pil_image = Image.fromarray(image)
    pil_image.save(buffer, 'jpeg')
    buffered_img = types.InputFile(buffer)
    buffer.seek(0)

    return buffered_img
