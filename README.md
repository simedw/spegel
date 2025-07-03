# Spegel – Reflect the web through AI

Automatically rewrites the websites into markdown optimised for viewing in the terminal.
Read intro blog post [here](https://simedw.com/2025/06/23/introducing-spegel/)

This is a proof-of-concept, bugs are to be expected but feel free to raise an issue or pull request.

##  Screenshot
Sometimes you don't want to read through someone's life story just to get to a recipe
![Recipe Example](https://simedw.com/2025/06/23/introducing-spegel/images/recipe_example.png)


## Installation

Requires Python 3.11+

### Basic installation
```bash
pip install spegel
```

### With AI providers
Choose your preferred AI provider(s):

```bash
# Install with OpenAI support
pip install "spegel[openai]"

# Install with Gemini support  
pip install "spegel[gemini]"

# Install with Claude support
pip install "spegel[claude]"

# Install with all ai providers
pip install "spegel[all]"
```

### From source
Clone the repo and install in editable mode: 

```bash
# Clone and enter the directory
$ git clone <repo-url>
$ cd spegel

# Install dependencies and the CLI
$ pip install -e .

# Or with specific providers
$ pip install -e ".[openai]"
$ pip install -e ".[gemini]" 
$ pip install -e ".[claude]"
$ pip install -e ".[all]"
```

## API Keys
Spegel supports multiple AI providers. Configure your API key(s):

### For Gemini
```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

### For OpenAI
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### For Claude
```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

You can configure which provider to use in your config file (see [Editing settings](#editing-settings) below).


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

### AI Provider Configuration

The `[ai]` section configures which AI provider to use:

- **`provider`**: Choose `"gemini"`, `"openai"`, or `"claude"`
- **`model`**: Model name (e.g., `"gemini-2.5-flash-lite-preview-06-17"` for Gemini, `"gpt-4.1-nano"` for OpenAI, `"claude-3-haiku-20240307"` for Claude)
- **`api_key_env`**: Environment variable containing your API key (`"GEMINI_API_KEY"`, `"OPENAI_API_KEY"`, or `"ANTHROPIC_API_KEY"`) (`DO NOT PUT YOUR API KEY HERE!`)
- **`temperature`**: Controls creativity (0.0 = deterministic, 2.0 = very creative)
- **`max_tokens`**: Maximum response length

Example snippet:
```toml
[ai]
provider = "gemini"
model = "gemini-2.5-flash-lite-preview-06-17"
api_key_env = "GEMINI_API_KEY"
temperature = 0.2
max_tokens = 8192

[settings]
default_view = "terminal"
app_title = "Spegel"

[[views]]
id = "raw"
name = "Raw View"
prompt = ""

[[views]]
id = "terminal"
name = "Terminal"
prompt = "Transform this webpage into the perfect terminal browsing experience! ..."
```

---

For more, see the code or open an issue!
