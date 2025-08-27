import asyncio
import logging
import os
import signal
import urllib3
import subprocess
import time
from collections import namedtuple
from datetime import datetime
from typing import Any
from pathlib import Path

import psutil
import requests
from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from minitrade.broker import BrokerAccount
from minitrade.utils.telegram import send_telegram_message

logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger("uvicorn.error")

__ib_loc = os.path.expanduser("~/.minitrade/ibgateway")


class GatewayStatus(BaseModel):
    account: str
    account_id: int
    pid: int
    port: int
    authenticated: bool
    connected: bool
    timestamp: str


GatewayInstance = namedtuple("GatewayInstance", ["pid", "port"])

app = FastAPI(title="IB gateway admin")


def ib_start():
    import uvicorn
    from minitrade.utils.config import config

    try:
        uvicorn.run(
            "minitrade.broker.ibgateway:app",
            host=config.brokers.ib.gateway_host,
            port=config.brokers.ib.gateway_port,
            log_level=config.brokers.ib.gateway_log_level,
        )
    except Exception as e:
        logger.error(f"Failed to start IB gateway: {e}")
        raise e


def call_ibgateway(
    instance: GatewayInstance,
    method: str,
    path: str,
    params: dict | None = None,
    timeout: int = 10,
) -> Any:
    """Call the ibgateway's admin API"""
    url = f"https://localhost:{instance.port}/v1/api{path}"
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        resp = requests.request(
            method=method, url=url, params=params, verify=False, timeout=timeout
        )
    except requests.exceptions.ConnectionError:
        return requests.exceptions.ConnectionError
    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code >= 400:
        raise RuntimeError(f"Request {path} returned {resp.status_code} {resp.text}")


def kill_all_ibgateway():
    """kill all running gateway instances"""
    for proc in psutil.process_iter():
        try:
            if proc.cmdline()[-1] == "ibgroup.web.core.clientportal.gw.GatewayStart":
                logger.info(proc)
                proc.kill()
        except Exception:
            pass


def test_ibgateway():
    """Test if IB gateway can be launched successfully

    Raises:
        RuntimeError: If IB gateway can't be launched successfully
    """
    pid, _ = launch_ibgateway()
    psutil.Process(pid).terminate()


def launch_ibgateway() -> GatewayInstance:
    """Launch IB gateway to listen on a random port

    Returns:
        instance: Return process id and port number if the gateway is successfully launched

    Raises:
        RuntimeError: If launching gateway failed
    """
    # def get_random_port():
    #     sock = socket.socket()
    #     sock.bind(('', 0))
    #     return sock.getsockname()[1]
    try:
        # port = get_random_port()
        port = 5000
        cmd = [
            "bash",
            "bin/run.sh",
            "root/conf.yaml",
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(os.getenv("ib_clientportal_folder")),
        )
        return GatewayInstance(proc.pid, port)
    except Exception as e:
        raise RuntimeError(f'Launching gateway instance failed: {" ".join(cmd)}') from e


def ping_ibgateway(username: str, instance: GatewayInstance) -> dict:
    """Get gateway connection status and kill gateway instance if corrupted

    Args:
        instance: The gateway instance

    Returns:
        pid: Gateway process ID
        port: Gateway listening port number
        account: IB account username
        account_id: IB account ID
        authenticated: If user is authenticated
        connected: If broker connection is established
        timestamp: Timestamp of status check

    Raises:
        HTTPException: 503 - If getting gateway status failed
    """
    try:
        tickle = call_ibgateway(instance, "GET", "tickle", timeout=5)
        logger.debug(f"{username} gateway tickle: {tickle}")
        sso = call_ibgateway(instance, "GET", "sso/validate", timeout=5)
        logger.debug(f"{username} gateway sso: {sso}")
        if sso and tickle:
            return {
                "pid": instance.pid,
                "port": instance.port,
                "account": sso["USER_NAME"],
                "account_id": tickle["userId"],
                "authenticated": tickle["iserver"]["authStatus"]["authenticated"],
                "connected": tickle["iserver"]["authStatus"]["connected"],
                "timestamp": datetime.now().isoformat(),
            }
        # If responses are missing, treat as failure
        raise HTTPException(503, f"IB ping error: {username}")
    except Exception as e:
        logger.debug(f"{username} gateway invalid, killing it")
        kill_ibgateway(username, instance)
        send_telegram_message(f"IB gateway disconnected: {username}, {e}")
        raise HTTPException(503, f"IB ping error: {username}") from e


# IB 2FA challenge response code
challenge_response = None


def login_ibgateway() -> dict:
    # Launch gateway and wait until auth endpoint responds
    instance = launch_ibgateway()
    logging.info(f"Visit https://localhost:{instance.port}/ to log in to IBKR")

    deadline = time.time() + 120
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            # Touch tickle and sso first to warm up
            try:
                call_ibgateway(instance, "GET", "tickle", timeout=3)
                call_ibgateway(instance, "GET", "sso/validate", timeout=3)
            except Exception:
                pass
            login_status = call_ibgateway(
                instance, "GET", "iserver/auth/status", timeout=5
            )
            if login_status:
                return login_status
        except Exception as e:
            last_error = e
        time.sleep(1)

    if last_error is not None:
        raise requests.exceptions.ConnectionError(
            f"Failed to connect to IB gateway: {last_error}"
        )
    raise requests.exceptions.ConnectionError("Failed to connect to IB gateway")


async def ibgateway_keepalive() -> None:
    """Keep gateway connections live, ping broker every 1 minute"""
    loop = asyncio.get_event_loop()
    while True:
        for username, instance in app.state.registry.copy().items():
            try:
                status = await loop.run_in_executor(
                    None, lambda: ping_ibgateway(username, instance)
                )
                logger.debug(f"{username} keepalive: {status}")
            except Exception as e:
                logger.error(e)
        await asyncio.sleep(60)


def kill_ibgateway(username: str, instance: GatewayInstance) -> None:
    """Kill gateway instance

    Args:
        instance: The gateway instance to kill
    """
    app.state.registry.pop(username, None)
    psutil.Process(instance.pid).terminate()
    logger.debug(f"{username} gateway killed")
    logger.debug(f"Gateway registry: {app.state.registry}")


@app.on_event("startup")
async def start_gateway():
    app.state.registry = {}
    asyncio.create_task(ibgateway_keepalive())


@app.on_event("shutdown")
async def shutdown_gateway():
    for username, instance in app.state.registry.copy().items():
        kill_ibgateway(username, instance)


def get_account(alias: str) -> BrokerAccount:
    account = BrokerAccount.get_account(alias)
    if account:
        return account
    else:
        raise HTTPException(404, f"Account {alias} not found")


@app.get("/ibgateway", response_model=list[GatewayStatus])
def get_gateway_status():
    """Return the gateway status"""
    status = []
    for username, inst in app.state.registry.copy().items():
        try:
            status.append(ping_ibgateway(username, inst))
        except Exception:
            pass
    return status


@app.get("/ibgateway/{alias}", response_model=GatewayStatus)
def get_account_status(account=Depends(get_account)):
    """Return the current gateway status associated with account `alias`

    Args:
        alias: Broker account alias

    Returns:
        200 with:
            pid: Gateway process ID
            port: Gateway listening port number
            account: IB account username
            account_id: IB account ID
            authenticated: If user is authenticated
            connected: If broker connection is established
            timestamp: Timestamp of status check
        or 204 if no gateway running.
    """
    instance = app.state.registry.get(account.username, None)
    if instance:
        return ping_ibgateway(account.username, instance)
    else:
        return Response(status_code=204)


@app.delete("/ibgateway/{alias}")
def exit_gateway(account=Depends(get_account)):
    """Exit a gateway instance that associates with account `alias`

    Args:
        alias: Broker account alias

    Returns:
        204
    """
    instance = app.state.registry.get(account.username, None)
    if instance:
        kill_ibgateway(account.username, instance)
    return Response(status_code=204)


@app.delete("/")
def exit_gateway_admin():
    """Exit all gateway instances and quit the app

    Returns:
        204
    """
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=204)


class ChallengeResponse(BaseModel):
    code: str


@app.post("/challenge")
def set_challenge_response(cr: ChallengeResponse):
    """Receive challenge response code

    Args:
        code: Challenge response code

    Returns:
        204
    """
    global challenge_response
    challenge_response = cr.code
    logger.debug(f"Challenge response received: {challenge_response}")
    return Response(status_code=204)
