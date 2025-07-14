# Spegel - Reflect the web through AI

Automatically rewrites the websites into markdown optimised for viewing in the terminal.
Read intro blog post [here](https://simedw.com/2025/06/23/introducing-spegel/)

This is a proof-of-concept, bugs are to be expected but feel free to raise an issue or pull request.

##  Screenshot
Sometimes you don't want to read through someone's life story just to get to a recipe
![Recipe Example](https://simedw.com/2025/06/23/introducing-spegel/images/recipe_example.png)


## Installation

Requires Python 3.11+

```bash
$ pip install spegel
```
or clone the repo and install it in editable mode

```bash
# Clone and enter the directory
$ git clone https://github.com/simedw/spegel.git
$ cd spegel

# Install dependencies and the CLI
$ pip install -e .
```

## API Keys
Spegel is using [litellm](https://github.com/BerriAI/litellm), which allows the use of the  common LLMs, both local and external. 

By default `Gemini 2.5 Flash Lite` is used, which requires you to set the `GEMINI_API_KEY`, see [env_example.txt](/env_example.txt)


## Usage

### Launch the browser

```bash
spegel                # Start with welcome screen
spegel bbc.com        # Open a URL immediately
```

Or, equivalently:

```bash
python -m spegel      # Start with welcome screen
python -m spegel bbc.com
```

### Basic controls
- `/`         – Open URL input
- `Tab`/`Shift+Tab` – Cycle links
- `Enter`     – Open selected link
- `e`         – Edit LLM prompt for current view
- `b`         – Go back
- `q`         – Quit

## Editing settings

Spegel loads settings from a TOML config file. You can customize views, prompts, and UI options.

**Config file search order:**
1. `./.spegel.toml` (current directory)
2. `~/.spegel.toml`
3. `~/.config/spegel/config.toml`

To edit settings:
1. Copy the example config:
   ```bash
   cp example_config.toml .spegel.toml
   # or create ~/.spegel.toml
   ```
2. Edit `.spegel.toml` in your favorite editor.

Example snippet:
```toml
[settings]
default_view = "terminal"
app_title = "Spegel"

[ai]
default_model="gpt-4.1-nano"

[[views]]
id = "raw"
name = "Raw View"
hotkey = "1"
order  = "1"
prompt = ""

[[views]]
id = "terminal"
name = "Terminal"
hotkey = "2"
order = "2"
prompt = "Transform this webpage into the perfect terminal browsing experience! ..."
model="claude-3-5-haiku-20241022"
```

## Local Models with Ollama

To run with a local model using Ollama, first pull and serve your desired model:

```bash
$ ollama pull llama2
$ ollama serve
```
Then set the model in `.spegel.toml` as follows:

```toml
model = "ollama/llama2"
```
Ollama supports models like Llama, Mistral, and many others.

## License
MIT License - see LICENSE file for details.


For more, see the code or open an issue!
