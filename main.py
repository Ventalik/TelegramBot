from aiogram import Bot, Dispatcher, executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from config import *

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot, storage=MemoryStorage())


async def shutdown(dispatcher: Dispatcher):
    await dispatcher.storage.close()
    await dispatcher.storage.wait_closed()


if __name__ == '__main__':
    from handlers import *
    executor.start_polling(dp, on_shutdown=shutdown)
