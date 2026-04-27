# PylaAI — Showdown Fork

This fork focuses on **Showdown** (trio). Other game modes still run off the upstream logic, but development effort and tuning here go into making Showdown play well end-to-end.

What the bot does in Showdown:

- **Analog joystick movement.** Brawlers are moved by a continuous angle, not WASD taps, so pathing and dodging are smoother than in the stock client-agnostic modes.
- **Follows teammates in trio** when there's no enemy to chase, with hysteresis so it doesn't ping-pong between two nearby teammates.
- **Passive roam** when alone and safe — slow rotation of standing still.
- **Poison fog avoidance.** Detects the fog and when a trusted fog mass enters the flee radius around the player, overrides movement to run the opposite way.
- **Wall-based unstuck detector + semicircle escape.** If surrounding walls stop moving while the bot is commanding movement, it's pressed against something — the bot retreats from the obstacle and then sweeps a semicircular arc around it. The arc side alternates between triggers.
- **Place-based trophy tracking.** Recognizes 1st/2nd/3rd/4th-place end screens and updates the trophy count accordingly.

---

PylaAI is currently the best external Brawl Stars bot.
This repository is intended for devs and it's recommended for others to use the official version from the discord.

**Warning :** This is the source-code, which is meant for developpers or people that know how to install python libraries and run python scripts --> The official build is linked in the discord, which is the source-code converted into an exe so you don't need additional knowledge to run the bot. (You will have to go through a linkvertise link)

How to run : 
- Install python and git(tested with python 3.11.9)
- open a cmd and type `git clone https://github.com/MrMuff1nn/PylaAI-OP.git`
- run `cd PylaAI-OP`
- run `python setup.py install`
- and then run `python main.py`
- Make sure your emulator is set to 1920x1080 (280dpi) in its settings
- enjoy !

- Additional Discord Integration
- 
Configure your Discord Bot:

    Go to the Discord Developer Portal and create a Bot.

    Turn on Message Content Intent in the Bot tab.

    Invite the bot to your private Discord server.

    Get your Discord User ID `(Right-click your profile in Discord -> Copy User ID).`

Update general_config.toml:
Open cfg/general_config.toml and add your Discord credentials at the bottom:
code Toml

discord_token = `"YOUR_DISCORD_BOT_TOKEN_HERE"`
discord_id = `"YOUR_DISCORD_USER_ID_HERE"`
personal_webhook = `"YOUR_DISCORD_WEBHOOK_URL_HERE"` # Optional: For goal completion alerts

Notes :
- This is the "localhost" version which means everything API related isn't enabled (login, online stats tracking, auto brawler list updating, auto icon updating, auto wall model updating). 
You can make it "online" by changing the base api url in utils.py and recoding the app to answer to the different endpoints. Site's code might become opensource but currently isn't.
- You can get the .pt version of the ai vision model at https://github.com/AngelFireLA/BrawlStarsBotMaking
- This repository won't contain early access features before they are released to the public.
- Please respect the "no selling" license as respect for our work.

Devs : 
- Iyordanov
- AngelFire


# Run tests
Run `python -m unittest discover` to check if your changes have made any regressions. 

# If you want to contribute, don't hesitate to create an Issue, a Pull Request, or/and make a ticket on the Pyla discord server at :
https://discord.gg/xUusk3fw4A

Don't know what to do ? Check the To-Fix and Idea lists :
https://trello.com/b/SAz9J6AA/public-pyla-trello
