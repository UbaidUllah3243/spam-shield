import struct
import numpy as np

class FlatBufferWriter:
 """Small reverse FlatBuffer schema writer for TFLite FULLY_CONNECTED/RELU/SOFTMAX."""
 def __init__(self): self.buf=bytearray();self.minalign=1;self.vtable=None;self.object_start=0
 def offset(self):return len(self.buf)
 def pad(self,n):self.buf[:0]=b'\0'*n
 def prep(self,size,additional=0):
  self.minalign=max(self.minalign,size);self.pad((-(len(self.buf)+additional))&(size-1))
 def prepend(self,fmt,v):
  sz=struct.calcsize('<'+fmt);self.prep(sz);self.buf[:0]=struct.pack('<'+fmt,v)
 def int32(self,v):self.prepend('i',v)
 def uint32(self,v):self.prepend('I',v)
 def byte(self,v):self.prepend('B',v)
 def float(self,v):self.prepend('f',v)
 def offset_to(self,target):
  self.prep(4);assert target<=self.offset();self.uint32(self.offset()-target+4)
 def vec_int(self,values):
  self.prep(4,len(values)*4);self.prep(4,len(values)*4)
  for v in reversed(values):self.int32(v)
  self.uint32(len(values));return self.offset()
 def vec_bytes(self,data):
  data=bytes(data);self.prep(4,len(data));self.prep(1,len(data))
  self.buf[:0]=data;self.uint32(len(data));return self.offset()
 def vec_offsets(self,offsets):
  self.prep(4,len(offsets)*4);self.prep(4,len(offsets)*4)
  for x in reversed(offsets):self.offset_to(x)
  self.uint32(len(offsets));return self.offset()
 def string(self,s):
  b=s.encode('utf8');self.prep(4,len(b)+1)
  self.buf[:0]=b'\0';self.buf[:0]=b;self.uint32(len(b));return self.offset()
 def start(self,n):
  assert self.vtable is None;self.vtable=[0]*n;self.object_start=self.offset()
 def slot_u(self,i,v,d=0):
  if v!=d:self.uint32(v);self.vtable[i]=self.offset()
 def slot_b(self,i,v,d=0):
  if v!=d:self.byte(v);self.vtable[i]=self.offset()
 def slot_f(self,i,v,d=0.):
  if v!=d:self.float(v);self.vtable[i]=self.offset()
 def slot_ref(self,i,off):
  if off:self.offset_to(off);self.vtable[i]=self.offset()
 def end(self):
  self.int32(0);obj=self.offset()
  for off in reversed(self.vtable):self.prepend('H',obj-off if off else 0)
  self.prepend('H',obj-self.object_start);self.prepend('H',(len(self.vtable)+2)*2)
  vt=self.offset();struct.pack_into('<i',self.buf,len(self.buf)-obj,vt-obj)
  self.vtable=None;return obj
 def finish(self,root):
  self.prep(self.minalign,8);self.buf[:0]=b'TFL3';self.offset_to(root);return bytes(self.buf)


def export_network(w0,b0,w1,b1,description):
    """Export a trained dense ReLU network into a TFLite schema-v3 FlatBuffer.
    Architecture: [1,N] => FullyConnected/ReLU => FullyConnected => Softmax [1,2].
    """
    assert w0.ndim==2 and b0.shape==(w0.shape[0],)
    assert w1.shape==(2,w0.shape[0]) and b1.shape==(2,)
    fb=FlatBufferWriter()
    buffers=[]
    for raw in [b'',w0.astype('<f4').tobytes(),b0.astype('<f4').tobytes(),
                w1.astype('<f4').tobytes(),b1.astype('<f4').tobytes()]:
        ref=fb.vec_bytes(raw) if raw else 0
        fb.start(1);fb.slot_ref(0,ref);buffers.append(fb.end())
    N=int(w0.shape[1]);H=int(w0.shape[0])
    desc=[([1,N],0,'message_features'),([H,N],1,'hidden_weights'),([H],2,'hidden_bias'),
       ([1,H],0,'hidden_relu'),([2,H],3,'output_weights'),([2],4,'output_bias'),
       ([1,2],0,'logits'),([1,2],0,'normal_spam_scores')]
    tensors=[]
    for shape,buffer_index,name in desc:
        shape_ref=fb.vec_int(shape);name_ref=fb.string(name)
        fb.start(12);fb.slot_ref(0,shape_ref);fb.slot_u(2,buffer_index);fb.slot_ref(3,name_ref)
        tensors.append(fb.end())
    fb.start(4);fb.slot_b(0,1);relu=fb.end() # ActivationFunctionType.RELU
    fb.start(4);fc_none=fb.end()
    fb.start(1);fb.slot_f(0,1.0);soft=fb.end()
    ops=[]
    for opcode,ins,outs,opt_type,opt in [(0,[0,1,2],[3],8,relu),
       (0,[3,4,5],[6],8,fc_none),(1,[6],[7],9,soft)]:
        insref=fb.vec_int(ins);outref=fb.vec_int(outs)
        fb.start(5);fb.slot_u(0,opcode);fb.slot_ref(1,insref);fb.slot_ref(2,outref)
        fb.slot_b(3,opt_type);fb.slot_ref(4,opt);ops.append(fb.end())
    ts=fb.vec_offsets(tensors);ins=fb.vec_int([0]);outs=fb.vec_int([7]);os=fb.vec_offsets(ops)
    name=fb.string('Spam Shield classification model')
    fb.start(5);fb.slot_ref(0,ts);fb.slot_ref(1,ins);fb.slot_ref(2,outs)
    fb.slot_ref(3,os);fb.slot_ref(4,name);graph=fb.end()
    codes=[]
    for opcode in [9,25]:
        fb.start(4);fb.slot_b(0,opcode);fb.slot_u(3,opcode);codes.append(fb.end())
    coderef=fb.vec_offsets(codes);graphs=fb.vec_offsets([graph]);bufref=fb.vec_offsets(buffers)
    comment=fb.string(description)
    fb.start(5);fb.slot_u(0,3);fb.slot_ref(1,coderef);fb.slot_ref(2,graphs)
    fb.slot_ref(3,comment);fb.slot_ref(4,bufref)
    blob=fb.finish(fb.end());assert blob[4:8]==b'TFL3'
    return blob
