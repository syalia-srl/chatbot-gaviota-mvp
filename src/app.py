import logging
from typing import Iterator, cast
from lingo import Message
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

from purely import ensure
from config import load
from beaver import BeaverDB
from bot import build
from lingo.core import Conversation

config = load()
db = BeaverDB(config.db)

# Setup logging (Crucial for debugging async apps)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class ConversationHandler(Conversation):
    def __init__(self, db: BeaverDB, user_id: int):
        self.db = db
        self.user_id = user_id
        self.list = db.list(f"conversation:{user_id}", model=Message)

    def append(self, message: Message, /):
        self.list.push(message)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.list)

    def __getitem__(self, index: int, /) -> Message:
        return cast(Message, self.list[index])

    def clear(self):
        self.list.clear()

    def __len__(self) -> int:
        return len(self.list)


# --- CORE HANDLERS (The "Upstream" Logic) ---


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Standard welcome message.
    """
    username = ensure(update.effective_user).first_name

    await ensure(update.message).reply_text(config.start.format(username=username))


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Standard welcome message.
    """
    username = ensure(update.effective_user).first_name
    user_id = ensure(update.effective_user).id

    conversation = ConversationHandler(db, user_id)
    chatbot = build(conversation)

    msg = ensure(update.effective_message).text

    if msg:
        reply = await chatbot.chat(msg)
        await ensure(update.effective_message).reply_text(reply.content)


# --- THE ENGINE STARTUP ---


def start_bot():
    """
    Main entry point. Loads config and starts the polling loop.
    """
    print("🚀 Building Application...")
    application = ApplicationBuilder().token(config.telegram.token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, chat))

    print("✅ Bot is running! Press Ctrl+C to stop.")
    # run_polling() handles the async event loop automatically
    application.run_polling()


if __name__ == "__main__":
    start_bot()
