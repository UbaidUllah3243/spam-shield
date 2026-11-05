"""Checks packaged model's TFLite schema, weight buffers and numerical consistency.
This is STATIC + NumPy verification, NOT a TensorFlow Lite Interpreter test.
"""
from pathlib import Path
import struct
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
b=(ROOT/'app/src/main/assets/spamshield.tflite').read_bytes()
weights=np.load(ROOT/'ml/weights.npz')
features=(ROOT/'app/src/main/assets/vocabulary.txt').read_text().strip().splitlines()

def ui(pos):return struct.unpack_from('<I',b,pos)[0]
def si(pos):return struct.unpack_from('<i',b,pos)[0]
def us(pos):return struct.unpack_from('<H',b,pos)[0]
def field(table,i):
    v=table-si(table)
    assert v >= 0 and v < len(b), 'Invalid vtable location'
    size=us(v)
    slot=v+4+i*2
    return table+us(slot) if slot < v+size and us(slot) else 0

def follow(pos):return pos+ui(pos) if pos else 0

def vec(table,slot):
    start=follow(field(table,slot))
    if not start:return []
    return [start+4+i*4 for i in range(ui(start))]

def offset_vec(table,slot):return [follow(x) for x in vec(table,slot)]

def int_vec(table,slot):return [si(x) for x in vec(table,slot)]

def byte_vec(table,slot):
    start=follow(field(table,slot))
    return b[start+4:start+4+ui(start)] if start else b''

assert b[4:8]==b'TFL3'
root=ui(0)
assert ui(field(root,0))==3, 'TFLite version must be 3'
subgraphs=offset_vec(root,2)
assert len(subgraphs)==1
sg=subgraphs[0]
tensors=offset_vec(sg,0)
shapes=[int_vec(t,0) for t in tensors]
assert shapes[0]==[1,len(features)], (shapes[0],len(features))
assert shapes[7]==[1,2]
assert shapes[1]==[40,len(features)] and shapes[4]==[2,40]
assert int_vec(sg,1)==[0] and int_vec(sg,2)==[7]
ops=offset_vec(sg,3)
assert len(ops)==3
assert [int_vec(o,1) for o in ops]==[[0,1,2],[3,4,5],[6]]
assert [int_vec(o,2) for o in ops]==[[3],[6],[7]]
codes=offset_vec(root,1)
assert [ui(field(c,3)) for c in codes]==[9,25], 'Expected FC/Softmax opcodes'
bufs=offset_vec(root,4)
for pos,key in [(1,'w0'),(2,'b0'),(3,'w1'),(4,'b1')]:
    actual=np.frombuffer(byte_vec(bufs[pos],0),dtype='<f4')
    expected=weights[key].reshape(-1)
    assert actual.shape==expected.shape and np.array_equal(actual,expected),key

x=np.zeros([1,len(features)],dtype=np.float32)
for token in ('claim','reward','free','urgent'):
 if token in features:x[0,features.index(token)]=1
w0=np.frombuffer(byte_vec(bufs[1],0),dtype='<f4').reshape(40,-1)
b0=np.frombuffer(byte_vec(bufs[2],0),dtype='<f4')
w1=np.frombuffer(byte_vec(bufs[3],0),dtype='<f4').reshape(2,-1)
b1=np.frombuffer(byte_vec(bufs[4],0),dtype='<f4')
h=np.maximum(x@w0.T+b0,0)
logits=h@w1.T+b1
scores=np.exp(logits-logits.max());scores/=scores.sum()
assert scores.shape==(1,2) and np.isfinite(scores).all() and abs(scores.sum()-1)<1e-6
print('OK: TFL3 FlatBuffer, schema version 3, 3 supported ops')
print('OK: input and output tensor shapes',shapes[0],shapes[7])
print('OK: all 4 binary weight arrays match training-exported weights')
print('OK: NumPy reference forward pass -> class scores',scores.round(4).tolist())
print('NOTE: Android Interpreter execution has not been tested here.')
