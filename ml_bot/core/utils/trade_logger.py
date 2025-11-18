import logging
import watchtower
import colorlog

timestamp_format = "%Y-%m-%d %H:%M:%S"

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Si les deux handlers sont déjà en place, on retourne immédiatement
    handler_types = {type(h).__name__ for h in logger.handlers}
    if "StreamHandler" in handler_types and "CloudWatchLogHandler" in handler_types:
        return logger

    # Console handler (coloré)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt=timestamp_format,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # CloudWatch handler (warnings only)
    cw_handler = watchtower.CloudWatchLogHandler(
        log_group="TradebotLogs",
        stream_name=name
    )
    cw_handler.setLevel(logging.WARNING)
    cw_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt=timestamp_format
    )
    cw_handler.setFormatter(cw_formatter)
    logger.addHandler(cw_handler)

    return logger

# 👇 Création de deux loggers indépendants
logger_backtest = setup_logger("logger_backtest")
logger_backtest.info("✅ All systems go.")
logger_backtest.warning("⚠️ Warning: Something might go sideways.")
logger_backtest.error("🔥 Boom.")

# 👇 DEBUG : Vérifie les handlers attachés à chaque logger
logger_backtest.debug(f"logger_pub has handlers: {[type(h).__name__ for h in logger_backtest.handlers]}")

def update_logger_levels(levels: dict):
    """Met à jour dynamiquement les niveaux de log (ex: {'logger_rest': 'INFO'})"""
    for name, level_str in levels.items():
        logger = logging.getLogger(name)
        try:
            level = getattr(logging, level_str.upper(), None)
            if level is not None:
                logger.setLevel(level)
                for handler in logger.handlers:
                    handler.setLevel(level)
                logger.info(f"🔄 Log level for '{name}' updated to {level_str.upper()}")
            else:
                logger.warning(f"⚠️ Invalid log level '{level_str}' for '{name}'")
        except Exception as e:
            logger.error(f"❌ Failed to update log level for {name}: {e}")


    
    