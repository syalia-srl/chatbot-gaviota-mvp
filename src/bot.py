from lingo import Lingo, LLM, Context, Engine
from config import load


config = load()

# Change name and description as desired to
# fit in the system prompt

chatbot = Lingo(
    name="Bot",
    description="a friendly chatbot",
    llm = LLM(**config.llm.model_dump()),

    # You can also modify the system prompt
    # to completely replace the chatbot personality.
    system_prompt=config.prompts.system,
)


# Add skills for the chatbot here
# Check out Lingo's documentation
# to learn how to write custom skills
# <https://github.com/gia-uh/lingo>

@chatbot.skill
async def chat(ctx: Context, engine: Engine):
    """Basic chat skill, just replies normally."""
    await engine.reply(ctx)
