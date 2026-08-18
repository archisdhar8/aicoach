from __future__ import annotations

import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

TELEGRAM_MAX_MESSAGE_LEN = 4096


@dataclass(frozen=True)
class SmsConfig:
    account_sid: str
    auth_token: str
    from_number: str
    to_number: str


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str


def load_sms_config() -> SmsConfig:
    values = {
        "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
        "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
        "from_number": os.getenv("TWILIO_FROM_NUMBER"),
        "to_number": os.getenv("ALERT_TO_NUMBER"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing SMS environment variables: "
            + ", ".join(missing)
            + ". Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, ALERT_TO_NUMBER."
        )
    return SmsConfig(**values)


def send_sms(message: str, config: SmsConfig | None = None) -> dict:
    config = config or load_sms_config()
    url = f"https://api.twilio.com/2010-04-01/Accounts/{config.account_sid}/Messages.json"
    response = requests.post(
        url,
        data={"From": config.from_number, "To": config.to_number, "Body": message},
        auth=(config.account_sid, config.auth_token),
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"Twilio SMS failed: status={response.status_code} body={response.text[:500]}")
    return response.json()


def load_telegram_config() -> TelegramConfig:
    values = {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError("Missing Telegram environment variables: " + ", ".join(missing))
    return TelegramConfig(**values)


def send_telegram(message: str, config: TelegramConfig | None = None) -> dict:
    config = config or load_telegram_config()
    url = f"https://api.telegram.org/bot{config.bot_token}/sendMessage"
    chunks = split_telegram_message(message)
    results: list[dict] = []
    try:
        for chunk in chunks:
            response = requests.post(
                url,
                data={"chat_id": config.chat_id, "text": chunk, "disable_web_page_preview": True},
                timeout=30,
            )
            if not response.ok:
                return _queue_telegram(
                    message,
                    error=f"Telegram send failed: status={response.status_code} body={response.text[:500]}",
                )
            payload = response.json()
            if isinstance(payload, dict):
                payload.setdefault("ok", True)
                results.append(payload)
            else:
                results.append({"ok": True, "result": payload})
        first_result = results[0] if results else {"ok": True, "result": {}}
        if len(results) <= 1:
            return first_result
        return {
            "ok": True,
            "result": first_result.get("result", {}),
            "messages": [item.get("result", {}) for item in results],
        }
    except requests.RequestException as exc:
        return _queue_telegram(message, error=f"Telegram send failed: {exc}")


def _queue_telegram(message: str, error: str) -> dict:
    outbox_root = Path(os.getenv("MLB_WINNERS_OUTBOX_DIR", "data/outbox"))
    outbox_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = outbox_root / f"telegram_{ts}.txt"
    path.write_text(message, encoding="utf-8")
    return {
        "ok": True,
        "queued": True,
        "error": error,
        "result": {"message_id": "queued", "outbox_path": str(path)},
    }


def split_telegram_message(message: str, max_length: int = TELEGRAM_MAX_MESSAGE_LEN) -> list[str]:
    if len(message) <= max_length:
        return [message]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in message.splitlines():
        fragments = _split_telegram_line(line, max_length=max_length)
        for fragment in fragments:
            separator_len = 1 if current else 0
            if current and current_len + separator_len + len(fragment) > max_length:
                chunks.append("\n".join(current))
                current = [fragment]
                current_len = len(fragment)
                continue
            if separator_len:
                current_len += separator_len
            current.append(fragment)
            current_len += len(fragment)

    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_telegram_line(line: str, max_length: int = TELEGRAM_MAX_MESSAGE_LEN) -> list[str]:
    if len(line) <= max_length:
        return [line]

    fragments: list[str] = []
    remaining = line
    while len(remaining) > max_length:
        split_at = remaining.rfind(" ", 0, max_length + 1)
        if split_at <= 0:
            split_at = max_length
        fragments.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        fragments.append(remaining)
    return fragments


def format_value_alert(predictions: pd.DataFrame, alert_date: str, max_rows: int = 6, empty_text: str | None = None) -> str:
    official = predictions[predictions["confidence"].isin(["strong", "medium"])].copy()
    watchlist = predictions[predictions["confidence"].eq("watchlist")].copy()
    if official.empty and watchlist.empty:
        return empty_text or f"MLB {alert_date}: no strong/medium/watchlist pregame plays right now."
    official = official.sort_values(["confidence", "ev_per_dollar"], ascending=[True, False]).head(max_rows)
    watchlist = watchlist.sort_values(["ev_per_dollar", "edge"], ascending=[False, False]).head(3)
    lines = [f"MLB {alert_date} value plays:"]
    if not official.empty:
        for row in official.to_dict("records"):
            lines.extend(_format_moneyline_alert_row(row))
    else:
        lines.append("No official strong/medium plays right now.")
    if not watchlist.empty:
        lines.append("")
        lines.append("WATCHLIST / LEANS - not official:")
        for row in watchlist.to_dict("records"):
            lines.extend(_format_moneyline_alert_row(row, label="LEAN"))
    return "\n".join(lines)


def _format_moneyline_alert_row(row: dict, label: str | None = None) -> list[str]:
    prob = float(row.get("bet_probability", 0.0)) * 100
    edge = float(row.get("edge", 0.0)) * 100
    ev = float(row.get("ev_per_dollar", 0.0))
    stake = float(row.get("stake_units", 1.0) or 0.0)
    line = int(row["bet_moneyline"]) if not pd.isna(row.get("bet_moneyline")) else "NA"
    heading = label or str(row.get("confidence", "")).upper()
    lines = [
        f"{heading}: {row['bet_side']} {line} "
        f"p={prob:.1f}% edge={edge:.1f}% EV={ev:.3f} stake={stake:.2f}u"
    ]
    risk_flags = row.get("risk_flags")
    if isinstance(risk_flags, str) and risk_flags.strip():
        lines.append(f"flags: {risk_flags.strip()}")
    reason = row.get("reason")
    if isinstance(reason, str) and reason.strip():
        lines.append(reason.strip())
    elif label == "LEAN":
        skip_reason = row.get("skip_reason")
        if isinstance(skip_reason, str) and skip_reason.strip():
            lines.append(f"lean reason: {skip_reason.strip()}")
    return lines


def filter_upcoming_predictions(
    predictions: pd.DataFrame,
    schedule: pd.DataFrame,
    window_minutes: int = 60,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    if predictions.empty or schedule.empty:
        return predictions
    status_cols = ["game_pk", "status", "game_datetime"]
    available = [column for column in status_cols if column in schedule.columns]
    slate = schedule[available].copy()
    merged = predictions.merge(slate, on="game_pk", how="left", suffixes=("", "_schedule"))
    if "status" not in merged.columns:
        return merged
    status = merged["status"].fillna("").str.lower()
    started = (
        status.str.contains("in progress")
        | status.str.contains("final")
        | status.str.contains("completed")
        | status.str.contains("game over")
        | status.str.contains("delayed")
    )
    upcoming = merged[~started].copy()
    if "game_datetime" not in upcoming.columns or upcoming["game_datetime"].isna().all():
        return upcoming.reset_index(drop=True)

    now_utc = now_utc or datetime.now(timezone.utc)
    start_times = pd.to_datetime(upcoming["game_datetime"], utc=True, errors="coerce")
    minutes_to_start = (start_times - pd.Timestamp(now_utc)).dt.total_seconds() / 60.0
    upcoming["minutes_to_start"] = minutes_to_start

    in_window = minutes_to_start.between(0, window_minutes, inclusive="both")
    data_ready = fully_ready(upcoming)
    return upcoming[in_window | data_ready].reset_index(drop=True)


def fully_ready(predictions: pd.DataFrame) -> pd.Series:
    ready = pd.Series(True, index=predictions.index)
    for column in ["home_probable_pitcher_id", "away_probable_pitcher_id", "bet_moneyline"]:
        if column in predictions.columns:
            ready &= predictions[column].notna()
    for column in ["home_starter_games_prior", "away_starter_games_prior"]:
        if column in predictions.columns:
            ready &= predictions[column].fillna(0).astype(float).ge(2)
    for column in ["home_lineup_confirmed", "away_lineup_confirmed"]:
        if column in predictions.columns:
            ready &= predictions[column].fillna(0).astype(float).ge(1)
    if "skip_reason" in predictions.columns:
        ready &= predictions["skip_reason"].isna() | predictions["skip_reason"].eq("")
    return ready
