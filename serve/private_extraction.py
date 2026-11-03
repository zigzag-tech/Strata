"""Opt-in private extraction: one bounded slot; uncertain work is quarantined.

No response history is stored. Configuration and tokens are never returned.
Native barrier and model identity still require real deployment qualification.
"""
import hmac
import json
import os
import re
import stat
import threading
import time

MAX_BYTES = 65536


class Refusal(RuntimeError):
    def __init__(self, status, cause, detail):
        self.status, self.cause, self.detail = status, cause, detail


def private_bytes(path, limit):
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError('private configuration requires absolute paths')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, os.getuid()) or info.st_mode & 0o077:
            raise ValueError('configuration must be an owned private regular file')
        value = stream.read(limit + 1)
        if len(value) > limit:
            raise ValueError('configuration exceeds its bound')
        return value


def load_runtime(svc, render, path='/etc/harmony/strata-extraction.json'):
    try:
        config = json.loads(private_bytes(path, 4096))
        if (set(config) != {'version','tokenFile','model','servedModel'}
                or type(config['version']) is not int or config['version'] != 1
                or config['model'] != 'flash_next' or config['servedModel'] != svc.model):
            raise ValueError('configuration identity differs from this service')
        token = private_bytes(config['tokenFile'], 256).decode().strip()
        if len(token) < 32 or any(c.isspace() for c in token):
            raise ValueError('private execution credential is invalid')
        return PrivateExtraction(svc, token, render)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError):
        print('[strata] protected extraction configuration refused', flush=True)
        return None


class PrivateExtraction:
    def __init__(self, svc, token, render):
        self.svc, self.token, self.render = svc, token, render
        svc.engine._harmony_extraction_context = threading.local()
        self.ingress = threading.Lock()
        self.active = None
        self.quarantined = False

    def handle(self, handler):
        identity, started, confirmed, acquired = '', False, False, False
        try:
            if handler.client_address[0] not in ('127.0.0.1','::1'):
                raise Refusal(403,'forbidden','private extraction requires loopback')
            presented = handler.headers.get('authorization','').encode()
            if not hmac.compare_digest(presented, ('Bearer '+self.token).encode()):
                raise Refusal(401,'unauthorized','private extraction credential required')
            nonce = handler.headers.get('x-harmony-execution-id','')
            if not re.fullmatch('[0-9a-f]{32}',nonce):
                raise Refusal(400,'invalid_input','bounded execution nonce required')
            identity = 'chatcmpl-'+nonce
            deadline = handler.headers.get('x-harmony-execution-deadline-ms','')
            if not re.fullmatch('[0-9]{13}',deadline):
                raise Refusal(400,'invalid_input','absolute execution deadline required')
            budget = (int(deadline)-int(time.time()*1000))/1000
            if budget <= 0:
                raise Refusal(504,'deadline_exceeded','execution already expired')
            if budget > 8:
                raise Refusal(400,'invalid_input','execution deadline exceeds eight seconds')
            expires = time.monotonic()+budget
            acquired = self.ingress.acquire(blocking=False)
            if not acquired or self.active is not None or self.quarantined:
                raise Refusal(429,'capacity','protected execution slot unavailable')
            size = handler.headers.get('content-length','')
            if not size.isdigit() or not 0 < int(size) <= MAX_BYTES or handler.headers.get('transfer-encoding'):
                raise Refusal(400,'invalid_input','bounded content length required')
            handler.connection.settimeout(max(0.001,expires-time.monotonic()))
            try:
                body=json.loads(handler.rfile.read(int(size)))
            except TimeoutError as exc:
                raise Refusal(504,'deadline_exceeded','execution body deadline exceeded') from exc
            except (ValueError,OSError) as exc:
                raise Refusal(400,'invalid_input','invalid or incomplete execution body') from exc
            expected={'model','stream','temperature','max_tokens','chat_template_kwargs','messages'}
            if (not isinstance(body,dict) or set(body)!=expected or body['model']!='flash_next'
                    or body['stream'] is not False or type(body['temperature']) not in (int,float)
                    or body['temperature']!=0 or type(body['max_tokens']) is not int or body['max_tokens']!=1024
                    or body['chat_template_kwargs']!={'enable_thinking':False}
                    or not isinstance(body['messages'],list) or len(body['messages'])!=2
                    or any(not isinstance(m,dict) or set(m)!={'role','content'} or m['role']!=role
                           or not isinstance(m['content'],str)
                           for m,role in zip(body['messages'],('system','user')))):
                raise Refusal(400,'invalid_input','fixed extraction policy required')
            if getattr(self.svc.engine,'batch',0):
                raise Refusal(503,'profile_unavailable','protected extraction requires exclusive single-sequence mode')
            if not getattr(self.svc.engine,'can_confirm_gpu_idle',False):
                raise Refusal(503,'profile_unavailable','native GPU barrier capability unavailable')
            if not self.svc.engine.alive():
                raise Refusal(503,'profile_unavailable','protected engine is not resident')
            if time.monotonic()>=expires:
                raise Refusal(504,'deadline_exceeded','execution expired before dispatch')
            if not self.svc.fifo.acquire(blocking=False):
                raise Refusal(429,'capacity','native generation slot occupied')
            cancel=threading.Event()
            done=threading.Event()
            result={}
            def barrier(engine):
                try:
                    from serve.bounded_stop import confirm_gpu_idle
                except ModuleNotFoundError:
                    from bounded_stop import confirm_gpu_idle
                confirm_gpu_idle(engine.proc,engine.lines,nonce,supported=engine.can_confirm_gpu_idle)
                result['confirmed']=True
            def worker():
                self.svc.engine._harmony_extraction_context.active = True
                try:
                    result['value']=self.render(body,cancel,barrier)
                except BaseException:
                    result['failed']=True
                finally:
                    self.svc.engine._harmony_extraction_context.active = False
                    if not result.get('confirmed'):
                        self.quarantined=True
                        self.svc.execution_quarantined=True
                    self.svc.fifo.release()
                    done.set()
            thread=threading.Thread(target=worker,daemon=True)
            self.active=thread
            started=True
            try:
                thread.start()
            except BaseException:
                self.active=None
                self.svc.fifo.release()
                started=False
                raise
            expired=not done.wait(max(0,expires-time.monotonic()))
            if expired:
                cancel.set()
                done.wait(4.5)
            confirmed=done.is_set() and result.get('confirmed') is True
            if not confirmed:
                self.quarantined=True
                raise Refusal(503,'unhealthy','execution cessation unconfirmed; slot retained')
            self.active=None
            if expired or time.monotonic()>=expires:
                raise Refusal(504,'deadline_exceeded','execution expired after confirmed cessation')
            if result.get('failed'):
                raise Refusal(422,'invalid_output','native extraction failed after confirmed cessation')
            value=result.get('value')
            if not isinstance(value,dict):
                raise Refusal(422,'invalid_output','native completion missing')
            value['id']=identity
            value['harmony_execution']={'request_id':identity,'completion_confirmed':True}
            if len(json.dumps(value).encode())>MAX_BYTES:
                raise Refusal(422,'invalid_output','completion exceeds 64 KiB')
            handler._json(200,value)
        except Refusal as exc:
            handler._json(exc.status,{'request_id':identity,'cause':exc.cause,'detail':exc.detail,
                'execution_started':started,'cessation_confirmed':confirmed})
        except Exception:
            handler._json(503,{'request_id':identity,'cause':'unhealthy','detail':'protected extraction failed',
                'execution_started':started,'cessation_confirmed':confirmed})
        finally:
            if acquired:
                self.ingress.release()
