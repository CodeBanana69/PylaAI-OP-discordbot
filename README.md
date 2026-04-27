🤖 PylaAI — Ultimate Showdown Fork (with Discord Integration)

This repository is a highly customized, advanced fork of PylaAI designed specifically to dominate in Showdown modes. It combines top-tier survival logic (poison fog detection, analog movement) with a fully interactive Discord Bot that allows you to monitor, pause, and control your bot remotely from your phone or PC.
✨ Key Features
🏆 Showdown Optimized Gameplay

    Advanced Movement: Analog movement mapping instead of standard 8-way directional clicks for smoother dodging.

    Survival Logic: Intelligent trio logic and poison fog evasion to maximize placements.

    Trophy Tracking: Automated stats tracking for Showdown placements (Win = 1st/2nd, Loss = 3rd/4th).

📱 Interactive Discord Control Panel

    Remote Control: Type !panel in your Discord server to bring up a control UI.

    Live Screenshots: Request a live frame of your emulator directly to Discord.

    Pause & Stop: Remotely freeze the bot (Pause) or kill the process entirely (Stop) if it gets stuck.

    Webhook Notifications: Get automatic notifications and screenshots when the bot finishes its brawler goals.

⚙️ Installation & Setup

    Clone the repository:
    code Bash

    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME

    Install Dependencies:
    Make sure you have Python installed, then install the required packages. (Note: discord.py is required for the remote control panel!)
    code Bash

    pip install -r requirements.txt
    pip install discord.py

    Configure your Discord Bot:

        Go to the Discord Developer Portal and create a Bot.

        Turn on Message Content Intent in the Bot tab.

        Invite the bot to your private Discord server.

        Get your Discord User ID (Right-click your profile in Discord -> Copy User ID).

    Update general_config.toml:
    Open cfg/general_config.toml and add your Discord credentials at the bottom:
    code Toml

    discord_token = "YOUR_DISCORD_BOT_TOKEN_HERE"
    discord_id = "YOUR_DISCORD_USER_ID_HERE"
    personal_webhook = "YOUR_DISCORD_WEBHOOK_URL_HERE" # Optional: For goal completion alerts

🚀 How to Run

    Open your BlueStacks / Emulator and launch Brawl Stars.

    Run the main script:
    code Bash

    python main.py

    The GUI will appear. Select your brawler and click Start.

    Discord Usage: Go to your Discord server and type !panel to summon the control panel. You can now minimize the terminal and monitor everything from Discord!

🙏 Credits & Acknowledgements

This fork wouldn't be possible without the incredible work of the open-source community. Huge thanks to:

    AngelFireLA & The PylaAI Team
    The Pioneers. Creators of the original PylaAI base. All core computer vision logic, ONNX models, window controllers, and base GUI structure belong to them.

    MrMuff1nn (PylaAI-OP)
    The Showdown Master. Developer of the advanced Showdown logic. Their fork introduced the analog movement, fog detection, and trio logic that makes this bot actually survive and win in Solo/Duo Showdown.

    myddxyz (BrawlIndustry)
    The Remote Controller. Creator of the interactive Discord integration. The asynchronous bot thread, shared state bridging, and !panel interactive UI were ported directly from their multi-instance fork.

⚠️ Disclaimer

This software is for educational purposes only. Using macros, automation, or bots violates the Supercell Terms of Service. Use this at your own risk; the creators and contributors of this repository are not responsible for any account bans or penalties.
