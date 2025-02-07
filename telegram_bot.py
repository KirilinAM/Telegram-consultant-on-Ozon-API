import os
from aiogram import Bot, Dispatcher, types, Router
from aiogram.filters import Command
from aiogram.types import Message
from search_ask import ask_on_ozon_api

# Создаем роутер
router = Router()

# Предполагаемая функция обработки (должна быть реализована)
# def ask(text: str) -> str:
#     return "Processed: " + text

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """Обработчик команды /start"""
    await message.answer("Бот запущен! Отправьте мне текст для обработки.")

@router.message()
async def handle_text(message: Message) -> None:
    """Обработчик текстовых сообщений"""
    try:
        # Передаем текст в функцию ask
        response = ask_on_ozon_api(message.text)
        await message.answer(response)
    except Exception as e:
        await message.answer(f"Произошла ошибка: {str(e)}")

async def main() -> None:
    """Запуск бота"""
    # Получаем токен из переменных среды
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Токен бота не найден в переменных окружения!")
    
    # Инициализация бота и диспетчера
    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(router)
    
    # Запускаем поллинг
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
