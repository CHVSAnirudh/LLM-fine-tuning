import logging
import inspect
from datetime import datetime
from typing import Optional

from asgi_correlation_id import CorrelationIdFilter
from rich.logging import RichHandler


class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        frame = inspect.currentframe().f_back
        while frame:
            if (
                frame.f_globals["__name__"] != __name__
                and frame.f_globals["__name__"] != "logging"
            ):
                break
            frame = frame.f_back

        if frame:
            self.filename = frame.f_code.co_filename
            file_names = self.filename.split("/")
            self.filename_short = (
                "/".join(file_names[-2:]) if len(file_names) > 1 else "unknown"
            )
            self.lineno = frame.f_lineno
        else:
            self.filename = "unknown"
            self.lineno = 0


class CustomRichHandler(RichHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_render.omit_repeated_times = False

    def render(
        self,
        *,
        record,
        traceback,
        message_renderable,
    ):
        """Render log for display.

        Args:
            record (LogRecord): logging Record.
            traceback (Optional[Traceback]): Traceback instance or None for no Traceback.
            message_renderable (ConsoleRenderable): Renderable (typically Text) containing log message contents.

        Returns:
            ConsoleRenderable: Renderable to display log.
        """
        # path = Path(record.pathname).name
        path = record.filename_short
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)

        log_renderable = self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )
        return log_renderable


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Logger(metaclass=SingletonMeta):
    def __init__(
        self,
        logger_name="Task Executor Automation",
        log_level=logging.DEBUG,
        use_rich: bool = False,
    ):
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(log_level)
            self.logger.makeRecord = self._make_custom_log_record
            self.logger.addFilter(CorrelationIdFilter(default_value="-"))

            if use_rich:
                formatter = logging.Formatter(
                    # "[grey50]%(name)s[/]   [bold yellow]%(correlation_id)s[/]   %(message)s",
                    " %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S %Z",
                )
                console_handler = CustomRichHandler(
                    level=log_level,
                    markup=True,
                    rich_tracebacks=True,
                    show_path=True,
                    tracebacks_show_locals=True,
                )
            else:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - [%(filename)s:%(lineno)d] - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S %Z",
                )
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)

            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _make_custom_log_record(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        return CustomLogRecord(
            name,
            level,
            fn,
            lno,
            msg,
            args,
            exc_info,
            func=func,
            extra=extra,
            sinfo=sinfo,
        )

    def debug(self, message, *args):
        self.logger.debug(message)
        if len(args) > 0:
            self.logger.debug(*args)

    def info(self, message, *args):
        self.logger.info(message)
        if len(args) > 0:
            self.logger.info(*args)

    def warning(self, message, *args):
        self.logger.warning(message)
        if len(args) > 0:
            self.logger.warning(*args)

    def error(self, message, exc_info=True, *args):
        self.logger.error(message, exc_info=exc_info)
        if len(args) > 0:
            self.logger.error(*args)

    def critical(self, message, exc_info=True, *args):
        self.logger.critical(message, exc_info=exc_info)
        if len(args) > 0:
            self.logger.critical(*args)


logger = Logger("automation.app", use_rich=True, log_level=logging.DEBUG)