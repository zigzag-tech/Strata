"""Bound cancellation drain; timeout never becomes a successful DONE receipt.

A confirmed process exit is process-lifecycle evidence, not GPU qualification.
"""
import queue
import subprocess
import time


class StopUnconfirmed(RuntimeError):
    pass


def stop_and_drain(proc, lines, *, can_stop, drain_seconds=2.0, exit_seconds=2.0, initial_lines=()):
    deadline = time.monotonic() + drain_seconds
    if can_stop:
        try:
            proc.stdin.write("STOP\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass
    pending = iter(initial_lines)
    while time.monotonic() < deadline:
        try:
            line = next(pending, ... )
            if line is ...:
                line = lines.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            break
        if line is not None and line.startswith("DONE "):
            return line
        if line is None or line.startswith("ERR"):
            break
    # Do not leave an unbounded drain behind or permit a new engine alongside
    # the old process. Even a successful kill is a failure of the DONE contract.
    try:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=exit_seconds)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StopUnconfirmed("STOP produced no DONE; engine exit remains unconfirmed") from exc
    raise StopUnconfirmed("STOP produced no DONE within its bound; engine process exited")


def confirm_gpu_idle(proc, lines, nonce, *, supported, timeout_seconds=2.0):
    """Caller MUST hold the generation FIFO until this correlated barrier ends."""
    if not supported:
        raise StopUnconfirmed("engine does not advertise a GPU barrier")
    if len(nonce) != 32 or any(c not in '0123456789abcdef' for c in nonce):
        raise ValueError("invalid barrier identity")
    try:
        proc.stdin.write('BARRIER ' + nonce + '\n')
        proc.stdin.flush()
        line = lines.get(timeout=timeout_seconds)
    except (OSError, ValueError, queue.Empty) as exc:
        raise StopUnconfirmed("GPU barrier acknowledgement unavailable") from exc
    if line is None or line.strip() != 'BARRIER ' + nonce + ' ok':
        raise StopUnconfirmed("GPU barrier acknowledgement failed or uncorrelated")
    return {'request_id': nonce, 'gpu_idle_confirmed': True}
