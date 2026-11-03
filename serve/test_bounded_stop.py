"""Real subprocess pipe/exit controls; no CUDA completion claim."""
import queue
import subprocess
import sys
import threading
import time

import pytest

from serve.bounded_stop import StopUnconfirmed, stop_and_drain


def process(mode):
    script = """import sys,time
mode=sys.argv[1]
line=sys.stdin.readline()
if mode=='done': print('DONE 0 1 0 0 cancel 0 0 0',flush=True)
elif mode=='error': print('ERR cancellation failed',flush=True)
time.sleep(60)
"""
    proc = subprocess.Popen([sys.executable, '-u', '-c', script, mode],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    lines = queue.Queue()
    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)
    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return proc, lines, thread


def cleanup(proc, thread):
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=2)
    thread.join(timeout=2)
    proc.stdin.close()
    proc.stdout.close()


def test_done_is_returned_without_killing_the_engine():
    proc, lines, thread = process('done')
    try:
        assert stop_and_drain(proc, lines, can_stop=True, drain_seconds=1).startswith('DONE ')
        assert proc.poll() is None
    finally:
        cleanup(proc, thread)


@pytest.mark.parametrize('mode,can_stop', [('ignore',True),('error',True),('ignore',False)])
def test_missing_done_is_bounded_and_never_reported_as_completion(mode,can_stop):
    proc, lines, thread = process(mode)
    try:
        started = time.monotonic()
        with pytest.raises(StopUnconfirmed, match='no DONE'):
            stop_and_drain(proc, lines, can_stop=can_stop, drain_seconds=0.1, exit_seconds=1)
        assert time.monotonic()-started < 2
        assert proc.poll() is not None
    finally:
        cleanup(proc, thread)


def test_strata_generator_closes_through_bounded_protocol(tmp_path):
    from serve.server import StrataEngine
    exe = tmp_path/'engine'
    exe.write_text('#!' + sys.executable + "\nimport sys,time\nprint('READY 4096 stop',flush=True)\nfor line in sys.stdin:\n if line.startswith('GEN '): print('T 7',flush=True)\n elif line.strip()=='STOP': print('DONE 1 1 0 0 cancel 0 0 0',flush=True)\n elif line.strip()=='QUIT': break\n")
    exe.chmod(0o700)
    engine = StrataEngine(str(exe), [])
    try:
        gen = engine.generate([1], 10, {}, threading.Event())
        assert next(gen) == 7
        gen.close()
        assert engine.last['finish'] == 'cancel'
        assert engine.proc.poll() is None
    finally:
        proc = engine.proc
        engine.close()
        proc.wait(timeout=2)


@pytest.mark.parametrize('mode', ['ok','failed','wrong','silent'])
def test_barrier_requires_a_correlated_native_acknowledgement(mode):
    from serve.bounded_stop import confirm_gpu_idle
    script = """import sys,time
nonce=sys.stdin.readline().strip().split()[1]
mode=sys.argv[1]
if mode!='silent': print('BARRIER '+('f'*32 if mode=='wrong' else nonce)+' '+('failed' if mode=='failed' else 'ok'),flush=True)
time.sleep(60)
"""
    proc=subprocess.Popen([sys.executable,'-u','-c',script,mode],stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,text=True)
    lines=queue.Queue()
    def pump():
        for line in proc.stdout: lines.put(line)
        lines.put(None)
    thread=threading.Thread(target=pump,daemon=True);thread.start()
    try:
        nonce='a'*32
        if mode=='ok':
            assert confirm_gpu_idle(proc,lines,nonce,supported=True,timeout_seconds=1)['request_id']==nonce
        else:
            with pytest.raises(StopUnconfirmed):
                confirm_gpu_idle(proc,lines,nonce,supported=True,timeout_seconds=0.2)
    finally:
        cleanup(proc,thread)


def test_old_engine_cannot_supply_protected_barrier():
    from serve.bounded_stop import confirm_gpu_idle
    with pytest.raises(StopUnconfirmed,match='does not advertise'):
        confirm_gpu_idle(None,None,'a'*32,supported=False)


def test_service_barrier_runs_before_generation_fifo_is_released():
    # This lock-order invariant is independent of GPU math; the existing server
    # MockEngine permits deterministic completion without a model pack.
    from serve.server import Service, MockEngine, ByteTokenizer
    from serve.frontend import ChatTemplate
    from pathlib import Path
    tok=ByteTokenizer()
    engine=MockEngine(tok,'hello',max_context=4096)
    svc=Service(engine,tok,ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
    observed=[]
    def stopped(actual_engine):
        assert actual_engine is engine
        assert svc.fifo.locked()
        observed.append('barrier')
    events=list(svc.run([1],False,False,20,{},threading.Event(),on_stopped=stopped))
    assert observed==['barrier']
    assert events[-1][0]=='done'
    assert not svc.fifo.locked()


def test_ordinary_chat_keeps_its_existing_stop_allowance(tmp_path):
    from serve.server import StrataEngine
    exe=tmp_path/'slow-stop-engine'
    exe.write_text('#!'+sys.executable+"\nimport sys,time\nprint('READY 4096 stop',flush=True)\nfor line in sys.stdin:\n if line.startswith('GEN '): print('T 7',flush=True)\n elif line.strip()=='STOP':\n  time.sleep(2.2)\n  print('DONE 1 1 0 0 cancel 0 0 0',flush=True)\n elif line.strip()=='QUIT': break\n")
    exe.chmod(0o700)
    engine=StrataEngine(str(exe),[])
    engine.silence_s=5
    proc=engine.proc
    try:
        gen=engine.generate([1],10,{},threading.Event())
        assert next(gen)==7
        gen.close()
        assert proc.poll() is None
        assert engine.last['finish']=='cancel'
    finally:
        engine.close();proc.wait(timeout=2)
