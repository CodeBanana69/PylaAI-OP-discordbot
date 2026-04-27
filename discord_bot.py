import discord
from discord.ext import commands
from utils import load_toml_as_dict, save_dict_as_toml
import asyncio
import time as _time
from brawl_industry_logger import get_logger, pop_logs
import cv2
import io
import sys

instance_id = "Main"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--instance" and _i + 1 < len(sys.argv):
        instance_id = sys.argv[_i + 1]
        break

log = get_logger("brawl_industry.discord")

def get_panel_text(bot_state) -> str:
    status = "Running" if bot_state.is_running() else "Stopped"
    return (
        f"Brawl Industry Control Panel — Instance {instance_id}\n"
        f"**Status:** {status}\n"
    )

class ControlPanelView(discord.ui.View):

    def __init__(self, bot_state):
        super().__init__(timeout=None)
        self.bot_state = bot_state

    @discord.ui.button(label="Start", style=discord.ButtonStyle.green, custom_id="bi:start")
    async def start_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.bot_state.is_running():
            await interaction.response.send_message("Bot is already running.", ephemeral=True)
            return
        self.bot_state.set_running(True)
        log.info("Bot STARTED by user")
        await interaction.response.edit_message(content=get_panel_text(self.bot_state), view=self)
        await interaction.followup.send("Bot started.", ephemeral=True)

    @discord.ui.button(label="Stop", style=discord.ButtonStyle.red, custom_id="bi:stop")
    async def stop_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.bot_state.is_running():
            await interaction.response.send_message("Bot is already stopped.", ephemeral=True)
            return
        self.bot_state.set_running(False)
        log.info("Bot STOPPED by user")
        await interaction.response.edit_message(content=get_panel_text(self.bot_state), view=self)
        await interaction.followup.send("Bot stopped.", ephemeral=True)

    @discord.ui.button(label="Restart game", style=discord.ButtonStyle.gray, custom_id="bi:restart")
    async def restart_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.bot_state.restart_func:
            await interaction.response.send_message("Restart not available yet.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        try:

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.bot_state.restart_func)
            log.info("Brawl Stars restarted by user")
            await interaction.followup.send("Brawl Stars restarted.", ephemeral=True)
        except Exception as e:
            log.error(f"Restart failed: {e}")
            await interaction.followup.send("Restart failed. Check logs.", ephemeral=True)

    @discord.ui.button(label="Screenshot", style=discord.ButtonStyle.blurple, custom_id="bi:screenshot")
    async def screenshot_button(self, interaction: discord.Interaction, button: discord.ui.Button):

        if not getattr(self.bot_state, "get_screenshot", None):
            await interaction.response.send_message("Screenshot engine not ready.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        try:
            frame, _, _ = self.bot_state.get_screenshot()
            if frame is None:
                await interaction.followup.send("Frame is empty.", ephemeral=True)
                return
            success, buffer = cv2.imencode(".png", frame)
            if not success:
                await interaction.followup.send("Encode failed.", ephemeral=True)
                return
            file = discord.File(io.BytesIO(buffer), filename="screenshot.png")
            await interaction.followup.send("Screenshot:", file=file, ephemeral=True)
            log.info("Screenshot sent to Discord.")
        except Exception as e:
            log.error(f"Screenshot failed: {e}")

            await interaction.followup.send("Screenshot failed. Check bot logs.", ephemeral=True)

def create_discord_bot(bot_state):

    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

    bot._notify_channel = None
    bot._log_channel    = None

    @bot.event
    async def on_ready():
        log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
        bot.add_view(ControlPanelView(bot_state))

        if not check_stuck_alert.is_running():
            check_stuck_alert.start()
        if not check_notifications.is_running():
            check_notifications.start()
        if not drain_logs.is_running():
            drain_logs.start()
        saved_id = load_toml_as_dict("cfg/general_config.toml").get("log_channel_id")
        if saved_id:
            for guild in bot.guilds:
                ch = guild.get_channel(int(saved_id))
                if ch:
                    bot._log_channel = ch
                    log.info(f"Log channel restored: #{ch.name}")
                    break

    @bot.event
    async def on_disconnect():
        log.warning("Discord connection lost. Reconnecting...")

    @bot.event
    async def on_resumed():
        log.info("Discord connection resumed.")

    async def _send_alert(message: str):
        if bot._notify_channel:
            try:
                await bot._notify_channel.send(message)
                return
            except Exception:
                bot._notify_channel = None

        for guild in bot.guilds:
            for channel in guild.text_channels:
                try:
                    if channel.permissions_for(guild.me).send_messages:
                        await channel.send(message)
                        bot._notify_channel = channel
                        return
                except Exception:
                    continue

    @commands.cooldown(1, 3, commands.BucketType.user)
    @bot.command(name="panel")
    async def panel_command(ctx: commands.Context):
        bot._notify_channel = ctx.channel
        view = ControlPanelView(bot_state)
        await ctx.send(get_panel_text(bot_state), view=view)

    @commands.cooldown(1, 3, commands.BucketType.user)
    @bot.command(name="bi_help")
    async def help_command(ctx: commands.Context):
        await ctx.send(
            "```\n"
            "Brawl Industry Commands\n"
            "─────────────────────\n"
            "!panel      → control panel (start/stop, game mode)\n"
            "!stats      → session stats\n"
            "!resetstats → reset wins/losses\n"
            "!setlogs    → stream console logs in this channel\n"
            "!bi_help    → this message\n"
            "```"
        )

    @commands.cooldown(1, 3, commands.BucketType.user)
    @bot.command(name="resetstats")
    async def reset_command(ctx: commands.Context):
        if bot_state.reset_stats_func:
            bot_state.reset_stats_func()
            await ctx.send("Stats reset to 0.")
        else:
            await ctx.send("Reset not available right now.")

    @commands.cooldown(1, 3, commands.BucketType.user)
    @bot.command(name="stats")
    async def stats_command(ctx: commands.Context):
        status      = "running" if bot_state.is_running() else "stopped"
        ips         = f"{bot_state.get_ips():.2f}" if bot_state.get_ips() else "..."
        stats       = bot_state.session_stats_func() if bot_state.session_stats_func else {}
        total_secs  = stats.get("time_played", 0.0)
        h, rem      = divmod(int(total_secs), 3600)
        m, s        = divmod(rem, 60)
        session_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

        await ctx.send(
            "```\n"
            "Session Stats\n"
            "─────────────────────\n"
            f"status    : {status}\n"
            f"brawler   : {bot_state.get_current_brawler() or '...'}\n"
            f"ips       : {ips}\n"
            f"session   : {session_str}\n"
            f"wins      : {stats.get('wins', 0)}\n"
            f"losses    : {stats.get('losses', 0)}\n"
            f"draws     : {stats.get('draws', 0)}\n"
            f"winrate   : {stats.get('winrate', 0.0):.1f}%\n"
            f"games     : {stats.get('games', 0)}\n"
            "```"
        )

    @commands.cooldown(1, 3, commands.BucketType.user)
    @bot.command(name="setlogs")
    async def setlogs_command(ctx: commands.Context):
        bot._log_channel = ctx.channel
        cfg = load_toml_as_dict("cfg/general_config.toml")
        cfg["log_channel_id"] = ctx.channel.id
        save_dict_as_toml(cfg, "cfg/general_config.toml")
        await ctx.send(f"Logs redirected to {ctx.channel.mention} (saved).", delete_after=5)
        log.info(f"Log stream set to #{ctx.channel.name} (id={ctx.channel.id})")

    from discord.ext import tasks

    @tasks.loop(seconds=2)
    async def drain_logs():
        ch = getattr(bot, "_log_channel", None)
        if not ch:
            return
        lines = pop_logs(max_lines=20)
        if not lines:
            return
        msg = "```\n" + "\n".join(lines) + "\n```"
        try:
            await ch.send(msg)
        except Exception:
            pass

    @tasks.loop(seconds=3)
    async def check_stuck_alert():
        if bot_state.get_stuck_alert():
            bot_state.set_stuck_alert(False)
            discord_id = load_toml_as_dict("cfg/general_config.toml").get("discord_id", "").strip()
            ping = f"<@{discord_id}> " if discord_id else ""
            await _send_alert(
                f"{ping}**[Brawl Industry {instance_id}] Alert:** Bot stuck — Brawl Stars could not restart. "
                "Press **Start** to resume after fixing your emulator."
            )

    @tasks.loop(seconds=10)
    async def check_notifications():
        msg = bot_state.pop_notification()
        while msg is not None:
            await _send_alert(f"**[Brawl Industry {instance_id}]** {msg}")
            msg = bot_state.pop_notification()

    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"wait {error.retry_after:.0f}s", delete_after=3)
        elif isinstance(error, commands.CheckFailure):
            pass
        else:
            raise error

    return bot
