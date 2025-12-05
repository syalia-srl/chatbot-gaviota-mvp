import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from purely import ensure
from config import load

config = load()

# Setup logging (Crucial for debugging async apps)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- CORE HANDLERS (The "Upstream" Logic) ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Standard welcome message.
    """
    username = ensure(update.effective_user).first_name

    await ensure(update.message).reply_text(
        config.start.format(username=username)
    )


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Standard welcome message.
    """
    username = ensure(update.effective_user).first_name

    await ensure(update.message).reply_text(
        "Nice"
    )

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
