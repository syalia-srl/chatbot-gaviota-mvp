import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from purely import ensure
from config import load

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
    user_first_name = ensure(update.effective_user).first_name
    await ensure(update.message).reply_text(
        f"Hello {user_first_name}! 👋\nI am a RAG Bot. Upload a PDF to start."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure(update.message).reply_text("Send me a document and I will ingest it.")

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fallback for commands we don't understand"""
    await ensure(update.message).reply_text("Sorry, I didn't understand that command.")

# --- THE ENGINE STARTUP ---

def start_bot():
    """
    Main entry point. Loads config and starts the polling loop.
    """
    print("🔌 Loading configuration...")
    config = load()

    print("🚀 Building Application...")
    # ApplicationBuilder is the modern, async way to build bots in PTB v20+
    application = ApplicationBuilder().token(config.telegram.token).build()

    # 1. Register Core Handlers (Upstream stuff)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # 2. Register Custom Skills (User's private logic)
    # This calls the function in src/app/custom_skills.py
    print("🔗 Hooking up custom skills...")

    # 3. Add Fallback (must be last)
    application.add_handler(MessageHandler(filters.COMMAND, unknown))

    print("✅ Bot is running! Press Ctrl+C to stop.")
    # run_polling() handles the async event loop automatically
    application.run_polling()


if __name__ == "__main__":
    start_bot()
