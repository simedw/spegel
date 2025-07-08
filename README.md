
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="right">
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=en">English</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=zh-CN">简体中文</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=zh-TW">繁體中文</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=ja">日本語</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=ko">한국어</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=hi">हिन्दी</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=th">ไทย</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=fr">Français</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=de">Deutsch</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=es">Español</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=it">Itapano</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=ru">Русский</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=pt">Português</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=nl">Nederlands</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=pl">Polski</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=ar">العربية</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=fa">فارسی</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=tr">Türkçe</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=vi">Tiếng Việt</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=simedw&project=spegel&lang=id">Bahasa Indonesia</a></p>
      </div>
    </div>
  </details>
</div>

# Spegel – Reflect the web through AI

Automatically rewrites the websites into markdown optimised for viewing in the terminal.
Read intro blog post [here](https://simedw.com/2025/06/23/introducing-spegel/)

This is a proof-of-concept, bugs are to be expected but feel free to raise an issue or pull request.

##  Screenshot
Sometimes you don't want to read through someone's life story just to get to a recipe
![Recipe Example](https://simedw.com/2025/06/23/introducing-spegel/images/recipe_example.png)


## Installation

Requires Python 3.11+

```
pip install spegel
```
or clone the repo and install it editable mode

```bash
# Clone and enter the directory
$ git clone <repo-url>
$ cd spegel

# Install dependencies and the CLI
$ pip install -e .
```

## API Keys
Spegel is currently only support Gemini 2.5 Flash, to use it you need to provide your API key in the env

```
GEMINI_API_KEY=...
```


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
