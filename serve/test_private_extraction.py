"""Actual HTTP/subprocess protocol; native shim is NOT GPU qualification."""
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import pytest

from serve.server import ByteTokenizer, Service, StrataEngine, serve
from serve.frontend import ChatTemplate

TOKEN='x'*48


@pytest.fixture
def endpoint(tmp_path):
    def start(mode='ok'):
        exe=tmp_path/('engine-'+mode)
        script="""import sys,json,time
mode=MODE
print('READY 4096 stop'+(' gpu_barrier' if mode!='old' else ''),flush=True)
for line in sys.stdin:
 if line.startswith('GEN '):
  if mode=='hang': time.sleep(60)
  if mode=='delayed': time.sleep(0.5)
  for t in b'{"outcome":"no_grounded_ask"}': print('T '+str(t),flush=True)
  print('T 257',flush=True)
  print('DONE 28 1 0 0 stop 0 0 0',flush=True)
 elif line.startswith('BARRIER '):
  print(line.strip()+(' failed' if mode=='failed' else ' ok'),flush=True)
 elif line.strip()=='QUIT': break
""".replace('MODE',repr(mode))
        exe.write_text('#!'+sys.executable+'\n'+script);exe.chmod(0o700)
        engine=StrataEngine(str(exe),[])
        svc=Service(engine,ByteTokenizer(),ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        token=tmp_path/'token';token.write_text(TOKEN);token.chmod(0o600)
        config=tmp_path/'config.json'
        config.write_text(json.dumps({'version':1,'tokenFile':str(token),'model':'flash_next','servedModel':svc.model}))
        config.chmod(0o600)
        httpd=serve(svc,port=0,private_execution_config=str(config))
        resources.append((httpd,engine))
        return 'http://127.0.0.1:'+str(httpd.server_address[1]),svc
    resources=[]
    yield start
    for httpd,engine in resources:
        proc=engine.proc
        httpd.shutdown();httpd.server_close();engine.close();proc.wait(timeout=2)


def call(base, *, token=TOKEN, change=None, headers=None):
    body={'model':'flash_next','stream':False,'temperature':0,'max_tokens':1024,
          'chat_template_kwargs':{'enable_thinking':False},
          'messages':[{'role':'system','content':'Extract only existing questions.'},
                      {'role':'user','content':'No question here.'}]}
    body.update(change or {})
    h={'content-type':'application/json','authorization':'Bearer '+token,
       'x-harmony-execution-id':'a'*32,
       'x-harmony-execution-deadline-ms':str(int(time.time()*1000)+8000)}
    h.update(headers or {})
    req=urllib.request.Request(base+'/harmony/extraction',data=json.dumps(body).encode(),headers=h)
    try:
        response=urllib.request.urlopen(req,timeout=15)
    except urllib.error.HTTPError as error:
        response=error
    with response: return response.status,json.loads(response.read(65537))


def test_correlated_completion_clears_the_single_slot(endpoint):
    base,svc=endpoint()
    status,body=call(base)
    assert status==200
    assert body['id']=='chatcmpl-'+'a'*32
    assert body['harmony_execution']=={'request_id':body['id'],'completion_confirmed':True}
    assert json.loads(body['choices'][0]['message']['content'])=={'outcome':'no_grounded_ask'}
    assert svc.private_extraction.active is None
    assert not svc.fifo.locked()


def test_auth_and_fixed_policy_refuse_before_execution(endpoint):
    base,svc=endpoint()
    assert call(base,token='wrong')[0]==401
    assert call(base,change={'max_tokens':2048})[0]==400
    assert call(base,headers={'x-harmony-execution-id':'bad'})[0]==400
    assert svc.private_extraction.active is None
    assert not svc.fifo.locked()


def test_old_native_engine_is_not_implicitly_qualified(endpoint):
    base,svc=endpoint('old')
    status,body=call(base)
    assert status==503 and body['execution_started'] is False
    assert body['cause']=='profile_unavailable'
    assert svc.private_extraction.active is None


def test_failed_barrier_quarantines_instead_of_releasing_as_complete(endpoint):
    base,svc=endpoint('failed')
    status,body=call(base)
    assert status==503
    assert body['execution_started'] is True and body['cessation_confirmed'] is False
    assert svc.private_extraction.quarantined
    assert svc.execution_quarantined
    assert call(base)[0]==429


def test_expired_request_cannot_start_execution(endpoint):
    base,svc=endpoint()
    status,body=call(base,headers={'x-harmony-execution-deadline-ms':str(int(time.time()*1000)-1)})
    assert status==504 and body['execution_started'] is False
    assert svc.private_extraction.active is None


def test_deadline_returns_only_after_correlated_cancellation_barrier(endpoint):
    base,svc=endpoint('delayed')
    status,body=call(base,headers={'x-harmony-execution-deadline-ms':str(int(time.time()*1000)+200)})
    assert status==504
    assert body['execution_started'] is True and body['cessation_confirmed'] is True
    assert svc.private_extraction.active is None
    assert not svc.fifo.locked()


def test_hung_native_execution_retains_uncertainty_after_bounded_failure(endpoint):
    base,svc=endpoint('hang')
    began=time.monotonic()
    status,body=call(base,headers={'x-harmony-execution-deadline-ms':str(int(time.time()*1000)+200)})
    assert status==503 and body['cessation_confirmed'] is False
    assert time.monotonic()-began < 7
    assert svc.private_extraction.quarantined
    assert call(base)[0]==429


def test_parallel_engine_cannot_claim_an_exclusive_device_barrier(endpoint):
    base,svc=endpoint()
    svc.engine.batch=2
    try:
        status,body=call(base)
        assert status==503 and body['execution_started'] is False
        assert svc.private_extraction.active is None
    finally:
        svc.engine.batch=0
