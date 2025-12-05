# Chatbot Template

A batteries-included, customizable starter pack for building chatbots. This project is designed to be forked and extended, providing a solid foundation with Telegram integration, LLM support, and database management.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

  * **Python 3.12+**
  * **uv**: An extremely fast Python package manager. (Install via `pip install uv` or see [docs.astral.sh/uv](https://docs.astral.sh/uv/)).
  * **Git**

You will also need:

1.  A **Telegram Bot Token** (Get this from [@BotFather](https://t.me/BotFather) on Telegram).
2.  An **LLM Provider API Key** (e.g., OpenRouter, OpenAI, Anthropic).

## 🚀 Quick Start

### 1. Fork and Clone

Fork this repository to your own GitHub account, then clone it to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/chatbot.git
cd chatbot
```

### 2. Configure the Environment

The project uses a YAML configuration file. You must create a local copy of the example configuration:

```bash
cp config.example.yaml config.yaml
```

**Important:** `config.yaml` is ignored by git to prevent accidental commitment of secrets.

### 3. Update Configuration

Open `config.yaml` in your editor. You have two options for setting up your credentials:

**Option A: Direct Entry (Simpler)**
Replace the `${...}` placeholders directly with your actual keys.

```yaml
telegram:
  token: "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"

llm:
  url: https://openrouter.ai/api/v1
  model: google/gemini-2.5-flash
  token: "sk-or-v1-..."
```

**Option B: Environment Variables (Recommended)**
Create a `.env` file in the root directory and add your secrets there. The configuration loader automatically substitutes values formatted as `${VAR_NAME}`.

1.  Create `.env`:
    ```bash
    TELEGRAM_TOKEN=your_telegram_token_here
    OPENROUTER_TOKEN=your_llm_token_here
    ```
2.  Keep `config.yaml` as is:
    ```yaml
    telegram:
      token: ${TELEGRAM_TOKEN}
    ...
    ```

### 4. Install Dependencies

Use `uv` to sync the project dependencies (this creates the virtual environment automatically):

```bash
uv sync
```

### 5. Run the Bot

You can start the bot using the provided Makefile or directly via `uv`:

**Using Make:**

```bash
make run
```

**Using UV directly:**

```bash
uv run src/bot.py
```

## 🛠 Project Structure

  * **`src/bot.py`**: The entry point. Handles Telegram updates and commands.
  * **`src/config.py`**: Handles loading YAML configuration and Pydantic validation.
  * **`config.yaml`**: Your local configuration (ignored by Git).
  * **`pyproject.toml`**: Defines dependencies (`beaver-db`, `lingo-ai`, `purely`, `python-telegram-bot`, etc.).

## 🧩 Extending the Bot

To add your own logic:

1.  Open `src/bot.py`.
2.  Define new `async` handler functions.
3.  Register them using `application.add_handler(...)` inside the `start_bot` function.

## 📄 License

This project is licensed under the **MIT License** - Copyright (c) 2025 Grupo de Inteligencia Artificial (GIA-UH).
