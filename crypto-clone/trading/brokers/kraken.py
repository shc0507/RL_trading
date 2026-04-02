"""Kraken spot REST integration."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import pandas as pd

from .base import AccountSnapshot, Broker, OrderRequest, OrderResult, PositionSnapshot


def _normalize_balance_code(asset_code: str) -> str:
    normalized = asset_code.split(".")[0].upper()
    if normalized in {"BTC", "XBT", "XXBT"}:
        return "BTC"
    if normalized.startswith("Z") and len(normalized) == 4:
        normalized = normalized[1:]
    if normalized.startswith("X") and len(normalized) == 4:
        normalized = normalized[1:]
    return normalized


def _base_asset_for_symbol(symbol: str, instrument_map: dict[str, dict[str, str]]) -> str:
    metadata = instrument_map.get(symbol, {})
    base_asset = metadata.get("base_asset")
    if base_asset:
        return str(base_asset).upper()
    return symbol.replace("/", "-").split("-")[0].upper()


def _broker_symbol_for(symbol: str, instrument_map: dict[str, dict[str, str]]) -> str:
    metadata = instrument_map.get(symbol, {})
    return str(metadata.get("broker_symbol", symbol)).upper()


class KrakenRESTClient:
    """Thin wrapper around Kraken's spot REST API."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.kraken.com") -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")

    def public_get(self, path: str, params: dict[str, object] | None = None) -> dict[str, object]:
        query = urllib.parse.urlencode(params or {})
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(url, method="GET")
        return self._read_json(request)

    def private_post(self, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        body = {key: value for key, value in (payload or {}).items()}
        body["nonce"] = str(int(time.time() * 1000))
        encoded = urllib.parse.urlencode(body)
        signature = self._sign(path=path, nonce=body["nonce"], post_data=encoded)
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=encoded.encode(),
            method="POST",
            headers={
                "API-Key": self.api_key,
                "API-Sign": signature,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        return self._read_json(request)

    def get_ticker_price(self, symbol: str) -> float:
        payload = self.public_get("/0/public/Ticker", {"pair": symbol})
        result = payload["result"]
        first_key = next(iter(result))
        return float(result[first_key]["c"][0])

    def get_balances(self) -> dict[str, float]:
        payload = self.private_post("/0/private/BalanceEx")
        result = payload["result"]
        normalized: dict[str, float] = {}
        for asset_code, amount in result.items():
            asset = _normalize_balance_code(asset_code)
            normalized[asset] = normalized.get(asset, 0.0) + float(amount)
        return normalized

    def add_market_order(self, symbol: str, side: str, quantity: float) -> dict[str, object]:
        return self.private_post(
            "/0/private/AddOrder",
            {
                "pair": symbol,
                "type": side,
                "ordertype": "market",
                "volume": f"{quantity:.8f}",
            },
        )

    def _sign(self, path: str, nonce: str, post_data: str) -> str:
        message = nonce.encode() + post_data.encode()
        sha256_hash = hashlib.sha256(message).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), path.encode() + sha256_hash, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _read_json(self, request: urllib.request.Request) -> dict[str, object]:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
        errors = payload.get("error", [])
        if errors:
            raise RuntimeError(f"kraken api error: {errors}")
        return payload


@dataclass(slots=True)
class KrakenBroker(Broker):
    """Spot broker backed by Kraken REST."""

    client: KrakenRESTClient
    quote_currency: str = "USD"
    instrument_map: dict[str, dict[str, str]] = field(default_factory=dict)
    name: str = "kraken"

    def get_account_snapshot(self, prices: dict[str, float]) -> AccountSnapshot:
        balances = self.client.get_balances()
        positions = self.get_positions(prices)
        equity = balances.get(self.quote_currency, 0.0) + sum(position.market_value for position in positions.values())
        return AccountSnapshot(
            timestamp=pd.Timestamp.now(tz="UTC").tz_localize(None),
            quote_currency=self.quote_currency,
            equity=float(equity),
            available_cash=float(balances.get(self.quote_currency, 0.0)),
            balances=dict(sorted(balances.items())),
        )

    def get_positions(self, prices: dict[str, float]) -> dict[str, PositionSnapshot]:
        balances = self.client.get_balances()
        positions: dict[str, PositionSnapshot] = {}
        for symbol, price in prices.items():
            base_asset = _base_asset_for_symbol(symbol, self.instrument_map)
            quantity = float(balances.get(base_asset, 0.0))
            if quantity <= 0.0:
                continue
            positions[symbol] = PositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                market_price=float(price),
                market_value=float(quantity * price),
            )
        return positions

    def place_market_order(self, order: OrderRequest, price_hint: float | None = None) -> OrderResult:
        broker_symbol = _broker_symbol_for(order.symbol, self.instrument_map)
        response = self.client.add_market_order(symbol=broker_symbol, side=order.side, quantity=order.quantity)
        average_price = float(price_hint or self.client.get_ticker_price(broker_symbol))
        return OrderResult(
            broker=self.name,
            symbol=order.symbol,
            side=order.side,
            quantity=float(order.quantity),
            average_price=average_price,
            filled_notional=float(order.quantity * average_price),
            fee=0.0,
            status="submitted",
            timestamp=pd.Timestamp.now(tz="UTC").tz_localize(None),
            raw_response=response,
        )
