from aiogram.utils.helper import Helper, HelperMode, Item


class BotStates(Helper):
    mode = HelperMode.snake_case

    START_STATE = Item()                # начальное состояние
    STYLE_TRANSFER_STATE_1 = Item()     # ожидание изображения стиля для NST
    STYLE_TRANSFER_STATE_2 = Item()     # ожидание изображения контента для NST
    STYLE_GAN_STATE = Item()            # ожидание выбора стиля для CycleGAN
    CUBISM_STATE = Item()               # ожидание изображения для CycleGAN
    RENAISSANCE_STATE = Item()          # ожидание изображения для CycleGAN
    EXPRESSIONISM_STATE = Item()        # ожидание изображения для CycleGAN
    SUPER_RESOLUTION_STATE = Item()     # ожидание изображения для ESRGAN

