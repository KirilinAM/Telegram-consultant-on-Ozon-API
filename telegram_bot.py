import os
from aiogram import Bot, Dispatcher, types, Router
from aiogram.filters import Command
from aiogram.types import Message
import asyncio
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

@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Обработчик команды /help"""
    help_text = (
        "Информация о базе знаний:\n\n"
        "📚 Тематика:\n"
        "Этот бот является консультантом по API Ozon. Он поможет вам разобраться "
        "с методами и возможностями платформы.\n\n"
        f"📊 Число записей в базе знаний: {len(df)}\n\n"
        "💡 Примеры запросов:\n"
        "- Какой метод получает информацию о товарах?\n"
        "- Какие возможности есть у Seller API?\n"
        "- Могу ли я управлять рекламой с помощью этого API? Опиши основные способы."
    )
    await message.answer(help_text)

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
    asyncio.run(main())
