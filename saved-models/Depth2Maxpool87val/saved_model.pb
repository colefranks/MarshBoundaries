??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02unknown8ܲ

?
conv2d_79/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_79/kernel
}
$conv2d_79/kernel/Read/ReadVariableOpReadVariableOpconv2d_79/kernel*&
_output_shapes
:*
dtype0
t
conv2d_79/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_79/bias
m
"conv2d_79/bias/Read/ReadVariableOpReadVariableOpconv2d_79/bias*
_output_shapes
:*
dtype0
?
conv2d_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameconv2d_80/kernel
}
$conv2d_80/kernel/Read/ReadVariableOpReadVariableOpconv2d_80/kernel*&
_output_shapes
:
*
dtype0
t
conv2d_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_80/bias
m
"conv2d_80/bias/Read/ReadVariableOpReadVariableOpconv2d_80/bias*
_output_shapes
:
*
dtype0
?
conv2d_transpose_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameconv2d_transpose_54/kernel
?
.conv2d_transpose_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_54/kernel*&
_output_shapes
:
*
dtype0
?
conv2d_transpose_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_54/bias
?
,conv2d_transpose_54/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_54/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_55/kernel
?
.conv2d_transpose_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_55/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_55/bias
?
,conv2d_transpose_55/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_55/bias*
_output_shapes
:*
dtype0
?
conv2d_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_81/kernel
}
$conv2d_81/kernel/Read/ReadVariableOpReadVariableOpconv2d_81/kernel*&
_output_shapes
:*
dtype0
t
conv2d_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_81/bias
m
"conv2d_81/bias/Read/ReadVariableOpReadVariableOpconv2d_81/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/conv2d_79/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_79/kernel/m
?
+Adam/conv2d_79/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_79/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_79/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_79/bias/m
{
)Adam/conv2d_79/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_79/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_80/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_80/kernel/m
?
+Adam/conv2d_80/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_80/kernel/m*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_80/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_80/bias/m
{
)Adam/conv2d_80/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_80/bias/m*
_output_shapes
:
*
dtype0
?
!Adam/conv2d_transpose_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!Adam/conv2d_transpose_54/kernel/m
?
5Adam/conv2d_transpose_54/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_54/kernel/m*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_transpose_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_54/bias/m
?
3Adam/conv2d_transpose_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_54/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_55/kernel/m
?
5Adam/conv2d_transpose_55/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_55/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_55/bias/m
?
3Adam/conv2d_transpose_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_55/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_81/kernel/m
?
+Adam/conv2d_81/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_81/bias/m
{
)Adam/conv2d_81/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_79/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_79/kernel/v
?
+Adam/conv2d_79/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_79/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_79/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_79/bias/v
{
)Adam/conv2d_79/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_79/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_80/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_80/kernel/v
?
+Adam/conv2d_80/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_80/kernel/v*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_80/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_80/bias/v
{
)Adam/conv2d_80/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_80/bias/v*
_output_shapes
:
*
dtype0
?
!Adam/conv2d_transpose_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!Adam/conv2d_transpose_54/kernel/v
?
5Adam/conv2d_transpose_54/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_54/kernel/v*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_transpose_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_54/bias/v
?
3Adam/conv2d_transpose_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_54/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_55/kernel/v
?
5Adam/conv2d_transpose_55/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_55/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_55/bias/v
?
3Adam/conv2d_transpose_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_55/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_81/kernel/v
?
+Adam/conv2d_81/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_81/bias/v
{
)Adam/conv2d_81/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?>
value?>B?> B?>
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures

_init_input_shape
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?
6iter

7beta_1

8beta_2
	9decay
:learning_ratemtmumvmw$mx%my*mz+m{0m|1m}v~vv?v?$v?%v?*v?+v?0v?1v?
F
0
1
2
3
$4
%5
*6
+7
08
19
F
0
1
2
3
$4
%5
*6
+7
08
19
 
?

;layers

	variables
<non_trainable_variables
=layer_metrics
>layer_regularization_losses
?metrics
trainable_variables
regularization_losses
 
 
\Z
VARIABLE_VALUEconv2d_79/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_79/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

@layers
	variables
Anon_trainable_variables
Blayer_metrics
Clayer_regularization_losses
Dmetrics
trainable_variables
regularization_losses
 
 
 
?

Elayers
	variables
Fnon_trainable_variables
Glayer_metrics
Hlayer_regularization_losses
Imetrics
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEconv2d_80/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_80/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

Jlayers
	variables
Knon_trainable_variables
Llayer_metrics
Mlayer_regularization_losses
Nmetrics
trainable_variables
regularization_losses
 
 
 
?

Olayers
 	variables
Pnon_trainable_variables
Qlayer_metrics
Rlayer_regularization_losses
Smetrics
!trainable_variables
"regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_54/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_54/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?

Tlayers
&	variables
Unon_trainable_variables
Vlayer_metrics
Wlayer_regularization_losses
Xmetrics
'trainable_variables
(regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_55/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_55/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
?

Ylayers
,	variables
Znon_trainable_variables
[layer_metrics
\layer_regularization_losses
]metrics
-trainable_variables
.regularization_losses
\Z
VARIABLE_VALUEconv2d_81/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_81/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
?

^layers
2	variables
_non_trainable_variables
`layer_metrics
alayer_regularization_losses
bmetrics
3trainable_variables
4regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
8
0
1
2
3
4
5
6
7
 
 
 

c0
d1
e2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ftotal
	gcount
h	variables
i	keras_api
D
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api
D
	ototal
	pcount
q
_fn_kwargs
r	variables
s	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

h	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

m	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

o0
p1

r	variables
}
VARIABLE_VALUEAdam/conv2d_79/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_79/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_80/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_80/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_54/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_54/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_55/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_55/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_81/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_81/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_79/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_79/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_80/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_80/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_54/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_54/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_55/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_55/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_81/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_81/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_37Placeholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_37conv2d_79/kernelconv2d_79/biasconv2d_80/kernelconv2d_80/biasconv2d_transpose_54/kernelconv2d_transpose_54/biasconv2d_transpose_55/kernelconv2d_transpose_55/biasconv2d_81/kernelconv2d_81/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_3914009
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_79/kernel/Read/ReadVariableOp"conv2d_79/bias/Read/ReadVariableOp$conv2d_80/kernel/Read/ReadVariableOp"conv2d_80/bias/Read/ReadVariableOp.conv2d_transpose_54/kernel/Read/ReadVariableOp,conv2d_transpose_54/bias/Read/ReadVariableOp.conv2d_transpose_55/kernel/Read/ReadVariableOp,conv2d_transpose_55/bias/Read/ReadVariableOp$conv2d_81/kernel/Read/ReadVariableOp"conv2d_81/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/conv2d_79/kernel/m/Read/ReadVariableOp)Adam/conv2d_79/bias/m/Read/ReadVariableOp+Adam/conv2d_80/kernel/m/Read/ReadVariableOp)Adam/conv2d_80/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_54/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_54/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_55/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_55/bias/m/Read/ReadVariableOp+Adam/conv2d_81/kernel/m/Read/ReadVariableOp)Adam/conv2d_81/bias/m/Read/ReadVariableOp+Adam/conv2d_79/kernel/v/Read/ReadVariableOp)Adam/conv2d_79/bias/v/Read/ReadVariableOp+Adam/conv2d_80/kernel/v/Read/ReadVariableOp)Adam/conv2d_80/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_54/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_54/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_55/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_55/bias/v/Read/ReadVariableOp+Adam/conv2d_81/kernel/v/Read/ReadVariableOp)Adam/conv2d_81/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_3914574
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_79/kernelconv2d_79/biasconv2d_80/kernelconv2d_80/biasconv2d_transpose_54/kernelconv2d_transpose_54/biasconv2d_transpose_55/kernelconv2d_transpose_55/biasconv2d_81/kernelconv2d_81/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/conv2d_79/kernel/mAdam/conv2d_79/bias/mAdam/conv2d_80/kernel/mAdam/conv2d_80/bias/m!Adam/conv2d_transpose_54/kernel/mAdam/conv2d_transpose_54/bias/m!Adam/conv2d_transpose_55/kernel/mAdam/conv2d_transpose_55/bias/mAdam/conv2d_81/kernel/mAdam/conv2d_81/bias/mAdam/conv2d_79/kernel/vAdam/conv2d_79/bias/vAdam/conv2d_80/kernel/vAdam/conv2d_80/bias/v!Adam/conv2d_transpose_54/kernel/vAdam/conv2d_transpose_54/bias/v!Adam/conv2d_transpose_55/kernel/vAdam/conv2d_transpose_55/bias/vAdam/conv2d_81/kernel/vAdam/conv2d_81/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_3914707??	
?
?
F__inference_conv2d_79_layer_call_and_return_conditional_losses_3913682

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?w
?	
E__inference_model_33_layer_call_and_return_conditional_losses_3914239

inputsB
(conv2d_79_conv2d_readvariableop_resource:7
)conv2d_79_biasadd_readvariableop_resource:B
(conv2d_80_conv2d_readvariableop_resource:
7
)conv2d_80_biasadd_readvariableop_resource:
V
<conv2d_transpose_54_conv2d_transpose_readvariableop_resource:
A
3conv2d_transpose_54_biasadd_readvariableop_resource:V
<conv2d_transpose_55_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_55_biasadd_readvariableop_resource:B
(conv2d_81_conv2d_readvariableop_resource:7
)conv2d_81_biasadd_readvariableop_resource:
identity?? conv2d_79/BiasAdd/ReadVariableOp?conv2d_79/Conv2D/ReadVariableOp? conv2d_80/BiasAdd/ReadVariableOp?conv2d_80/Conv2D/ReadVariableOp? conv2d_81/BiasAdd/ReadVariableOp?conv2d_81/Conv2D/ReadVariableOp?*conv2d_transpose_54/BiasAdd/ReadVariableOp?3conv2d_transpose_54/conv2d_transpose/ReadVariableOp?*conv2d_transpose_55/BiasAdd/ReadVariableOp?3conv2d_transpose_55/conv2d_transpose/ReadVariableOp?
conv2d_79/Conv2D/ReadVariableOpReadVariableOp(conv2d_79_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_79/Conv2DConv2Dinputs'conv2d_79/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
 conv2d_79/BiasAdd/ReadVariableOpReadVariableOp)conv2d_79_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_79/BiasAddBiasAddconv2d_79/Conv2D:output:0(conv2d_79/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????~
conv2d_79/ReluReluconv2d_79/BiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
max_pooling2d_28/MaxPoolMaxPoolconv2d_79/Relu:activations:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
conv2d_80/Conv2D/ReadVariableOpReadVariableOp(conv2d_80_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0?
conv2d_80/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_80/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
*
paddingSAME*
strides
?
 conv2d_80/BiasAdd/ReadVariableOpReadVariableOp)conv2d_80_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
conv2d_80/BiasAddBiasAddconv2d_80/Conv2D:output:0(conv2d_80/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
~
conv2d_80/ReluReluconv2d_80/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
?
max_pooling2d_29/MaxPoolMaxPoolconv2d_80/Relu:activations:0*A
_output_shapes/
-:+???????????????????????????
*
ksize
*
paddingSAME*
strides
j
conv2d_transpose_54/ShapeShape!max_pooling2d_29/MaxPool:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_54/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_54/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_54/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_54/strided_sliceStridedSlice"conv2d_transpose_54/Shape:output:00conv2d_transpose_54/strided_slice/stack:output:02conv2d_transpose_54/strided_slice/stack_1:output:02conv2d_transpose_54/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)conv2d_transpose_54/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_54/strided_slice_1StridedSlice"conv2d_transpose_54/Shape:output:02conv2d_transpose_54/strided_slice_1/stack:output:04conv2d_transpose_54/strided_slice_1/stack_1:output:04conv2d_transpose_54/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)conv2d_transpose_54/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_54/strided_slice_2StridedSlice"conv2d_transpose_54/Shape:output:02conv2d_transpose_54/strided_slice_2/stack:output:04conv2d_transpose_54/strided_slice_2/stack_1:output:04conv2d_transpose_54/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose_54/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_54/mulMul,conv2d_transpose_54/strided_slice_1:output:0"conv2d_transpose_54/mul/y:output:0*
T0*
_output_shapes
: [
conv2d_transpose_54/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_54/addAddV2conv2d_transpose_54/mul:z:0"conv2d_transpose_54/add/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_54/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_54/mul_1Mul,conv2d_transpose_54/strided_slice_2:output:0$conv2d_transpose_54/mul_1/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_54/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_54/add_1AddV2conv2d_transpose_54/mul_1:z:0$conv2d_transpose_54/add_1/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_54/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_54/stackPack*conv2d_transpose_54/strided_slice:output:0conv2d_transpose_54/add:z:0conv2d_transpose_54/add_1:z:0$conv2d_transpose_54/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_54/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_54/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_54/strided_slice_3StridedSlice"conv2d_transpose_54/stack:output:02conv2d_transpose_54/strided_slice_3/stack:output:04conv2d_transpose_54/strided_slice_3/stack_1:output:04conv2d_transpose_54/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_54/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_54_conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype0?
$conv2d_transpose_54/conv2d_transposeConv2DBackpropInput"conv2d_transpose_54/stack:output:0;conv2d_transpose_54/conv2d_transpose/ReadVariableOp:value:0!max_pooling2d_29/MaxPool:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
?
*conv2d_transpose_54/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_54/BiasAddBiasAdd-conv2d_transpose_54/conv2d_transpose:output:02conv2d_transpose_54/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
conv2d_transpose_54/ReluRelu$conv2d_transpose_54/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????o
conv2d_transpose_55/ShapeShape&conv2d_transpose_54/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_55/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_55/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_55/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_55/strided_sliceStridedSlice"conv2d_transpose_55/Shape:output:00conv2d_transpose_55/strided_slice/stack:output:02conv2d_transpose_55/strided_slice/stack_1:output:02conv2d_transpose_55/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)conv2d_transpose_55/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_55/strided_slice_1StridedSlice"conv2d_transpose_55/Shape:output:02conv2d_transpose_55/strided_slice_1/stack:output:04conv2d_transpose_55/strided_slice_1/stack_1:output:04conv2d_transpose_55/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)conv2d_transpose_55/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_55/strided_slice_2StridedSlice"conv2d_transpose_55/Shape:output:02conv2d_transpose_55/strided_slice_2/stack:output:04conv2d_transpose_55/strided_slice_2/stack_1:output:04conv2d_transpose_55/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose_55/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_55/mulMul,conv2d_transpose_55/strided_slice_1:output:0"conv2d_transpose_55/mul/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_55/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_55/mul_1Mul,conv2d_transpose_55/strided_slice_2:output:0$conv2d_transpose_55/mul_1/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_55/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_55/stackPack*conv2d_transpose_55/strided_slice:output:0conv2d_transpose_55/mul:z:0conv2d_transpose_55/mul_1:z:0$conv2d_transpose_55/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_55/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_55/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_55/strided_slice_3StridedSlice"conv2d_transpose_55/stack:output:02conv2d_transpose_55/strided_slice_3/stack:output:04conv2d_transpose_55/strided_slice_3/stack_1:output:04conv2d_transpose_55/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_55/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_55_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
$conv2d_transpose_55/conv2d_transposeConv2DBackpropInput"conv2d_transpose_55/stack:output:0;conv2d_transpose_55/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_54/Relu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
*conv2d_transpose_55/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_55/BiasAddBiasAdd-conv2d_transpose_55/conv2d_transpose:output:02conv2d_transpose_55/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
conv2d_81/Conv2D/ReadVariableOpReadVariableOp(conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_81/Conv2DConv2D$conv2d_transpose_55/BiasAdd:output:0'conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
 conv2d_81/BiasAdd/ReadVariableOpReadVariableOp)conv2d_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_81/BiasAddBiasAddconv2d_81/Conv2D:output:0(conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????~
conv2d_81/ReluReluconv2d_81/BiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityconv2d_81/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp!^conv2d_79/BiasAdd/ReadVariableOp ^conv2d_79/Conv2D/ReadVariableOp!^conv2d_80/BiasAdd/ReadVariableOp ^conv2d_80/Conv2D/ReadVariableOp!^conv2d_81/BiasAdd/ReadVariableOp ^conv2d_81/Conv2D/ReadVariableOp+^conv2d_transpose_54/BiasAdd/ReadVariableOp4^conv2d_transpose_54/conv2d_transpose/ReadVariableOp+^conv2d_transpose_55/BiasAdd/ReadVariableOp4^conv2d_transpose_55/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 2D
 conv2d_79/BiasAdd/ReadVariableOp conv2d_79/BiasAdd/ReadVariableOp2B
conv2d_79/Conv2D/ReadVariableOpconv2d_79/Conv2D/ReadVariableOp2D
 conv2d_80/BiasAdd/ReadVariableOp conv2d_80/BiasAdd/ReadVariableOp2B
conv2d_80/Conv2D/ReadVariableOpconv2d_80/Conv2D/ReadVariableOp2D
 conv2d_81/BiasAdd/ReadVariableOp conv2d_81/BiasAdd/ReadVariableOp2B
conv2d_81/Conv2D/ReadVariableOpconv2d_81/Conv2D/ReadVariableOp2X
*conv2d_transpose_54/BiasAdd/ReadVariableOp*conv2d_transpose_54/BiasAdd/ReadVariableOp2j
3conv2d_transpose_54/conv2d_transpose/ReadVariableOp3conv2d_transpose_54/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_55/BiasAdd/ReadVariableOp*conv2d_transpose_55/BiasAdd/ReadVariableOp2j
3conv2d_transpose_55/conv2d_transpose/ReadVariableOp3conv2d_transpose_55/conv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_80_layer_call_fn_3914288

inputs!
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_3913705?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_3914707
file_prefix;
!assignvariableop_conv2d_79_kernel:/
!assignvariableop_1_conv2d_79_bias:=
#assignvariableop_2_conv2d_80_kernel:
/
!assignvariableop_3_conv2d_80_bias:
G
-assignvariableop_4_conv2d_transpose_54_kernel:
9
+assignvariableop_5_conv2d_transpose_54_bias:G
-assignvariableop_6_conv2d_transpose_55_kernel:9
+assignvariableop_7_conv2d_transpose_55_bias:=
#assignvariableop_8_conv2d_81_kernel:/
!assignvariableop_9_conv2d_81_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: %
assignvariableop_19_total_2: %
assignvariableop_20_count_2: E
+assignvariableop_21_adam_conv2d_79_kernel_m:7
)assignvariableop_22_adam_conv2d_79_bias_m:E
+assignvariableop_23_adam_conv2d_80_kernel_m:
7
)assignvariableop_24_adam_conv2d_80_bias_m:
O
5assignvariableop_25_adam_conv2d_transpose_54_kernel_m:
A
3assignvariableop_26_adam_conv2d_transpose_54_bias_m:O
5assignvariableop_27_adam_conv2d_transpose_55_kernel_m:A
3assignvariableop_28_adam_conv2d_transpose_55_bias_m:E
+assignvariableop_29_adam_conv2d_81_kernel_m:7
)assignvariableop_30_adam_conv2d_81_bias_m:E
+assignvariableop_31_adam_conv2d_79_kernel_v:7
)assignvariableop_32_adam_conv2d_79_bias_v:E
+assignvariableop_33_adam_conv2d_80_kernel_v:
7
)assignvariableop_34_adam_conv2d_80_bias_v:
O
5assignvariableop_35_adam_conv2d_transpose_54_kernel_v:
A
3assignvariableop_36_adam_conv2d_transpose_54_bias_v:O
5assignvariableop_37_adam_conv2d_transpose_55_kernel_v:A
3assignvariableop_38_adam_conv2d_transpose_55_bias_v:E
+assignvariableop_39_adam_conv2d_81_kernel_v:7
)assignvariableop_40_adam_conv2d_81_bias_v:
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_79_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_79_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_80_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_80_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp-assignvariableop_4_conv2d_transpose_54_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_conv2d_transpose_54_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv2d_transpose_55_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv2d_transpose_55_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_81_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_81_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_79_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_79_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_80_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_80_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp5assignvariableop_25_adam_conv2d_transpose_54_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_conv2d_transpose_54_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_conv2d_transpose_55_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_conv2d_transpose_55_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_81_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_81_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_79_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_79_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_80_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_80_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_conv2d_transpose_54_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_conv2d_transpose_54_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_conv2d_transpose_55_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_conv2d_transpose_55_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_81_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_81_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
? 
?
P__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_3913657

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?&
?
E__inference_model_33_layer_call_and_return_conditional_losses_3913976
input_37+
conv2d_79_3913948:
conv2d_79_3913950:+
conv2d_80_3913954:

conv2d_80_3913956:
5
conv2d_transpose_54_3913960:
)
conv2d_transpose_54_3913962:5
conv2d_transpose_55_3913965:)
conv2d_transpose_55_3913967:+
conv2d_81_3913970:
conv2d_81_3913972:
identity??!conv2d_79/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall?+conv2d_transpose_54/StatefulPartitionedCall?+conv2d_transpose_55/StatefulPartitionedCall?
!conv2d_79/StatefulPartitionedCallStatefulPartitionedCallinput_37conv2d_79_3913948conv2d_79_3913950*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_79_layer_call_and_return_conditional_losses_3913682?
 max_pooling2d_28/PartitionedCallPartitionedCall*conv2d_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3913692?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_28/PartitionedCall:output:0conv2d_80_3913954conv2d_80_3913956*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_3913705?
 max_pooling2d_29/PartitionedCallPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3913715?
+conv2d_transpose_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_29/PartitionedCall:output:0conv2d_transpose_54_3913960conv2d_transpose_54_3913962*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_3913613?
+conv2d_transpose_55/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_54/StatefulPartitionedCall:output:0conv2d_transpose_55_3913965conv2d_transpose_55_3913967*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_3913657?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_55/StatefulPartitionedCall:output:0conv2d_81_3913970conv2d_81_3913972*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_3913738?
IdentityIdentity*conv2d_81/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp"^conv2d_79/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall,^conv2d_transpose_54/StatefulPartitionedCall,^conv2d_transpose_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 2F
!conv2d_79/StatefulPartitionedCall!conv2d_79/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2Z
+conv2d_transpose_54/StatefulPartitionedCall+conv2d_transpose_54/StatefulPartitionedCall2Z
+conv2d_transpose_55/StatefulPartitionedCall+conv2d_transpose_55/StatefulPartitionedCall:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
input_37
?#
?
P__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_3913613

inputsB
(conv2d_transpose_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????

 
_user_specified_nameinputs
?w
?	
E__inference_model_33_layer_call_and_return_conditional_losses_3914149

inputsB
(conv2d_79_conv2d_readvariableop_resource:7
)conv2d_79_biasadd_readvariableop_resource:B
(conv2d_80_conv2d_readvariableop_resource:
7
)conv2d_80_biasadd_readvariableop_resource:
V
<conv2d_transpose_54_conv2d_transpose_readvariableop_resource:
A
3conv2d_transpose_54_biasadd_readvariableop_resource:V
<conv2d_transpose_55_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_55_biasadd_readvariableop_resource:B
(conv2d_81_conv2d_readvariableop_resource:7
)conv2d_81_biasadd_readvariableop_resource:
identity?? conv2d_79/BiasAdd/ReadVariableOp?conv2d_79/Conv2D/ReadVariableOp? conv2d_80/BiasAdd/ReadVariableOp?conv2d_80/Conv2D/ReadVariableOp? conv2d_81/BiasAdd/ReadVariableOp?conv2d_81/Conv2D/ReadVariableOp?*conv2d_transpose_54/BiasAdd/ReadVariableOp?3conv2d_transpose_54/conv2d_transpose/ReadVariableOp?*conv2d_transpose_55/BiasAdd/ReadVariableOp?3conv2d_transpose_55/conv2d_transpose/ReadVariableOp?
conv2d_79/Conv2D/ReadVariableOpReadVariableOp(conv2d_79_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_79/Conv2DConv2Dinputs'conv2d_79/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
 conv2d_79/BiasAdd/ReadVariableOpReadVariableOp)conv2d_79_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_79/BiasAddBiasAddconv2d_79/Conv2D:output:0(conv2d_79/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????~
conv2d_79/ReluReluconv2d_79/BiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
max_pooling2d_28/MaxPoolMaxPoolconv2d_79/Relu:activations:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
conv2d_80/Conv2D/ReadVariableOpReadVariableOp(conv2d_80_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0?
conv2d_80/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_80/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
*
paddingSAME*
strides
?
 conv2d_80/BiasAdd/ReadVariableOpReadVariableOp)conv2d_80_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
conv2d_80/BiasAddBiasAddconv2d_80/Conv2D:output:0(conv2d_80/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
~
conv2d_80/ReluReluconv2d_80/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
?
max_pooling2d_29/MaxPoolMaxPoolconv2d_80/Relu:activations:0*A
_output_shapes/
-:+???????????????????????????
*
ksize
*
paddingSAME*
strides
j
conv2d_transpose_54/ShapeShape!max_pooling2d_29/MaxPool:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_54/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_54/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_54/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_54/strided_sliceStridedSlice"conv2d_transpose_54/Shape:output:00conv2d_transpose_54/strided_slice/stack:output:02conv2d_transpose_54/strided_slice/stack_1:output:02conv2d_transpose_54/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)conv2d_transpose_54/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_54/strided_slice_1StridedSlice"conv2d_transpose_54/Shape:output:02conv2d_transpose_54/strided_slice_1/stack:output:04conv2d_transpose_54/strided_slice_1/stack_1:output:04conv2d_transpose_54/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)conv2d_transpose_54/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_54/strided_slice_2StridedSlice"conv2d_transpose_54/Shape:output:02conv2d_transpose_54/strided_slice_2/stack:output:04conv2d_transpose_54/strided_slice_2/stack_1:output:04conv2d_transpose_54/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose_54/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_54/mulMul,conv2d_transpose_54/strided_slice_1:output:0"conv2d_transpose_54/mul/y:output:0*
T0*
_output_shapes
: [
conv2d_transpose_54/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_54/addAddV2conv2d_transpose_54/mul:z:0"conv2d_transpose_54/add/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_54/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_54/mul_1Mul,conv2d_transpose_54/strided_slice_2:output:0$conv2d_transpose_54/mul_1/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_54/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_54/add_1AddV2conv2d_transpose_54/mul_1:z:0$conv2d_transpose_54/add_1/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_54/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_54/stackPack*conv2d_transpose_54/strided_slice:output:0conv2d_transpose_54/add:z:0conv2d_transpose_54/add_1:z:0$conv2d_transpose_54/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_54/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_54/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_54/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_54/strided_slice_3StridedSlice"conv2d_transpose_54/stack:output:02conv2d_transpose_54/strided_slice_3/stack:output:04conv2d_transpose_54/strided_slice_3/stack_1:output:04conv2d_transpose_54/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_54/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_54_conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype0?
$conv2d_transpose_54/conv2d_transposeConv2DBackpropInput"conv2d_transpose_54/stack:output:0;conv2d_transpose_54/conv2d_transpose/ReadVariableOp:value:0!max_pooling2d_29/MaxPool:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
?
*conv2d_transpose_54/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_54/BiasAddBiasAdd-conv2d_transpose_54/conv2d_transpose:output:02conv2d_transpose_54/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
conv2d_transpose_54/ReluRelu$conv2d_transpose_54/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????o
conv2d_transpose_55/ShapeShape&conv2d_transpose_54/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_55/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_55/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_55/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_55/strided_sliceStridedSlice"conv2d_transpose_55/Shape:output:00conv2d_transpose_55/strided_slice/stack:output:02conv2d_transpose_55/strided_slice/stack_1:output:02conv2d_transpose_55/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)conv2d_transpose_55/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_55/strided_slice_1StridedSlice"conv2d_transpose_55/Shape:output:02conv2d_transpose_55/strided_slice_1/stack:output:04conv2d_transpose_55/strided_slice_1/stack_1:output:04conv2d_transpose_55/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)conv2d_transpose_55/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_55/strided_slice_2StridedSlice"conv2d_transpose_55/Shape:output:02conv2d_transpose_55/strided_slice_2/stack:output:04conv2d_transpose_55/strided_slice_2/stack_1:output:04conv2d_transpose_55/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose_55/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_55/mulMul,conv2d_transpose_55/strided_slice_1:output:0"conv2d_transpose_55/mul/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_55/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_55/mul_1Mul,conv2d_transpose_55/strided_slice_2:output:0$conv2d_transpose_55/mul_1/y:output:0*
T0*
_output_shapes
: ]
conv2d_transpose_55/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_55/stackPack*conv2d_transpose_55/strided_slice:output:0conv2d_transpose_55/mul:z:0conv2d_transpose_55/mul_1:z:0$conv2d_transpose_55/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_55/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_55/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_55/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_55/strided_slice_3StridedSlice"conv2d_transpose_55/stack:output:02conv2d_transpose_55/strided_slice_3/stack:output:04conv2d_transpose_55/strided_slice_3/stack_1:output:04conv2d_transpose_55/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_55/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_55_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
$conv2d_transpose_55/conv2d_transposeConv2DBackpropInput"conv2d_transpose_55/stack:output:0;conv2d_transpose_55/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_54/Relu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
*conv2d_transpose_55/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_55/BiasAddBiasAdd-conv2d_transpose_55/conv2d_transpose:output:02conv2d_transpose_55/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
conv2d_81/Conv2D/ReadVariableOpReadVariableOp(conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_81/Conv2DConv2D$conv2d_transpose_55/BiasAdd:output:0'conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
 conv2d_81/BiasAdd/ReadVariableOpReadVariableOp)conv2d_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_81/BiasAddBiasAddconv2d_81/Conv2D:output:0(conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????~
conv2d_81/ReluReluconv2d_81/BiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityconv2d_81/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp!^conv2d_79/BiasAdd/ReadVariableOp ^conv2d_79/Conv2D/ReadVariableOp!^conv2d_80/BiasAdd/ReadVariableOp ^conv2d_80/Conv2D/ReadVariableOp!^conv2d_81/BiasAdd/ReadVariableOp ^conv2d_81/Conv2D/ReadVariableOp+^conv2d_transpose_54/BiasAdd/ReadVariableOp4^conv2d_transpose_54/conv2d_transpose/ReadVariableOp+^conv2d_transpose_55/BiasAdd/ReadVariableOp4^conv2d_transpose_55/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 2D
 conv2d_79/BiasAdd/ReadVariableOp conv2d_79/BiasAdd/ReadVariableOp2B
conv2d_79/Conv2D/ReadVariableOpconv2d_79/Conv2D/ReadVariableOp2D
 conv2d_80/BiasAdd/ReadVariableOp conv2d_80/BiasAdd/ReadVariableOp2B
conv2d_80/Conv2D/ReadVariableOpconv2d_80/Conv2D/ReadVariableOp2D
 conv2d_81/BiasAdd/ReadVariableOp conv2d_81/BiasAdd/ReadVariableOp2B
conv2d_81/Conv2D/ReadVariableOpconv2d_81/Conv2D/ReadVariableOp2X
*conv2d_transpose_54/BiasAdd/ReadVariableOp*conv2d_transpose_54/BiasAdd/ReadVariableOp2j
3conv2d_transpose_54/conv2d_transpose/ReadVariableOp3conv2d_transpose_54/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_55/BiasAdd/ReadVariableOp*conv2d_transpose_55/BiasAdd/ReadVariableOp2j
3conv2d_transpose_55/conv2d_transpose/ReadVariableOp3conv2d_transpose_55/conv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_79_layer_call_and_return_conditional_losses_3914259

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3913556

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_80_layer_call_and_return_conditional_losses_3913705

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_3914009
input_37!
unknown:
	unknown_0:#
	unknown_1:

	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_37unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_3913547?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
input_37
??
?

"__inference__wrapped_model_3913547
input_37K
1model_33_conv2d_79_conv2d_readvariableop_resource:@
2model_33_conv2d_79_biasadd_readvariableop_resource:K
1model_33_conv2d_80_conv2d_readvariableop_resource:
@
2model_33_conv2d_80_biasadd_readvariableop_resource:
_
Emodel_33_conv2d_transpose_54_conv2d_transpose_readvariableop_resource:
J
<model_33_conv2d_transpose_54_biasadd_readvariableop_resource:_
Emodel_33_conv2d_transpose_55_conv2d_transpose_readvariableop_resource:J
<model_33_conv2d_transpose_55_biasadd_readvariableop_resource:K
1model_33_conv2d_81_conv2d_readvariableop_resource:@
2model_33_conv2d_81_biasadd_readvariableop_resource:
identity??)model_33/conv2d_79/BiasAdd/ReadVariableOp?(model_33/conv2d_79/Conv2D/ReadVariableOp?)model_33/conv2d_80/BiasAdd/ReadVariableOp?(model_33/conv2d_80/Conv2D/ReadVariableOp?)model_33/conv2d_81/BiasAdd/ReadVariableOp?(model_33/conv2d_81/Conv2D/ReadVariableOp?3model_33/conv2d_transpose_54/BiasAdd/ReadVariableOp?<model_33/conv2d_transpose_54/conv2d_transpose/ReadVariableOp?3model_33/conv2d_transpose_55/BiasAdd/ReadVariableOp?<model_33/conv2d_transpose_55/conv2d_transpose/ReadVariableOp?
(model_33/conv2d_79/Conv2D/ReadVariableOpReadVariableOp1model_33_conv2d_79_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_33/conv2d_79/Conv2DConv2Dinput_370model_33/conv2d_79/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
)model_33/conv2d_79/BiasAdd/ReadVariableOpReadVariableOp2model_33_conv2d_79_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_33/conv2d_79/BiasAddBiasAdd"model_33/conv2d_79/Conv2D:output:01model_33/conv2d_79/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
model_33/conv2d_79/ReluRelu#model_33/conv2d_79/BiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
!model_33/max_pooling2d_28/MaxPoolMaxPool%model_33/conv2d_79/Relu:activations:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
(model_33/conv2d_80/Conv2D/ReadVariableOpReadVariableOp1model_33_conv2d_80_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0?
model_33/conv2d_80/Conv2DConv2D*model_33/max_pooling2d_28/MaxPool:output:00model_33/conv2d_80/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
*
paddingSAME*
strides
?
)model_33/conv2d_80/BiasAdd/ReadVariableOpReadVariableOp2model_33_conv2d_80_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
model_33/conv2d_80/BiasAddBiasAdd"model_33/conv2d_80/Conv2D:output:01model_33/conv2d_80/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
?
model_33/conv2d_80/ReluRelu#model_33/conv2d_80/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
?
!model_33/max_pooling2d_29/MaxPoolMaxPool%model_33/conv2d_80/Relu:activations:0*A
_output_shapes/
-:+???????????????????????????
*
ksize
*
paddingSAME*
strides
|
"model_33/conv2d_transpose_54/ShapeShape*model_33/max_pooling2d_29/MaxPool:output:0*
T0*
_output_shapes
:z
0model_33/conv2d_transpose_54/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_33/conv2d_transpose_54/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_33/conv2d_transpose_54/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*model_33/conv2d_transpose_54/strided_sliceStridedSlice+model_33/conv2d_transpose_54/Shape:output:09model_33/conv2d_transpose_54/strided_slice/stack:output:0;model_33/conv2d_transpose_54/strided_slice/stack_1:output:0;model_33/conv2d_transpose_54/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
2model_33/conv2d_transpose_54/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_54/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_54/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,model_33/conv2d_transpose_54/strided_slice_1StridedSlice+model_33/conv2d_transpose_54/Shape:output:0;model_33/conv2d_transpose_54/strided_slice_1/stack:output:0=model_33/conv2d_transpose_54/strided_slice_1/stack_1:output:0=model_33/conv2d_transpose_54/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
2model_33/conv2d_transpose_54/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_54/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_54/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,model_33/conv2d_transpose_54/strided_slice_2StridedSlice+model_33/conv2d_transpose_54/Shape:output:0;model_33/conv2d_transpose_54/strided_slice_2/stack:output:0=model_33/conv2d_transpose_54/strided_slice_2/stack_1:output:0=model_33/conv2d_transpose_54/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_33/conv2d_transpose_54/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
 model_33/conv2d_transpose_54/mulMul5model_33/conv2d_transpose_54/strided_slice_1:output:0+model_33/conv2d_transpose_54/mul/y:output:0*
T0*
_output_shapes
: d
"model_33/conv2d_transpose_54/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
 model_33/conv2d_transpose_54/addAddV2$model_33/conv2d_transpose_54/mul:z:0+model_33/conv2d_transpose_54/add/y:output:0*
T0*
_output_shapes
: f
$model_33/conv2d_transpose_54/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
"model_33/conv2d_transpose_54/mul_1Mul5model_33/conv2d_transpose_54/strided_slice_2:output:0-model_33/conv2d_transpose_54/mul_1/y:output:0*
T0*
_output_shapes
: f
$model_33/conv2d_transpose_54/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
"model_33/conv2d_transpose_54/add_1AddV2&model_33/conv2d_transpose_54/mul_1:z:0-model_33/conv2d_transpose_54/add_1/y:output:0*
T0*
_output_shapes
: f
$model_33/conv2d_transpose_54/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
"model_33/conv2d_transpose_54/stackPack3model_33/conv2d_transpose_54/strided_slice:output:0$model_33/conv2d_transpose_54/add:z:0&model_33/conv2d_transpose_54/add_1:z:0-model_33/conv2d_transpose_54/stack/3:output:0*
N*
T0*
_output_shapes
:|
2model_33/conv2d_transpose_54/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4model_33/conv2d_transpose_54/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_54/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,model_33/conv2d_transpose_54/strided_slice_3StridedSlice+model_33/conv2d_transpose_54/stack:output:0;model_33/conv2d_transpose_54/strided_slice_3/stack:output:0=model_33/conv2d_transpose_54/strided_slice_3/stack_1:output:0=model_33/conv2d_transpose_54/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<model_33/conv2d_transpose_54/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_33_conv2d_transpose_54_conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype0?
-model_33/conv2d_transpose_54/conv2d_transposeConv2DBackpropInput+model_33/conv2d_transpose_54/stack:output:0Dmodel_33/conv2d_transpose_54/conv2d_transpose/ReadVariableOp:value:0*model_33/max_pooling2d_29/MaxPool:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
?
3model_33/conv2d_transpose_54/BiasAdd/ReadVariableOpReadVariableOp<model_33_conv2d_transpose_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$model_33/conv2d_transpose_54/BiasAddBiasAdd6model_33/conv2d_transpose_54/conv2d_transpose:output:0;model_33/conv2d_transpose_54/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
!model_33/conv2d_transpose_54/ReluRelu-model_33/conv2d_transpose_54/BiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
"model_33/conv2d_transpose_55/ShapeShape/model_33/conv2d_transpose_54/Relu:activations:0*
T0*
_output_shapes
:z
0model_33/conv2d_transpose_55/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_33/conv2d_transpose_55/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_33/conv2d_transpose_55/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*model_33/conv2d_transpose_55/strided_sliceStridedSlice+model_33/conv2d_transpose_55/Shape:output:09model_33/conv2d_transpose_55/strided_slice/stack:output:0;model_33/conv2d_transpose_55/strided_slice/stack_1:output:0;model_33/conv2d_transpose_55/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
2model_33/conv2d_transpose_55/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_55/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_55/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,model_33/conv2d_transpose_55/strided_slice_1StridedSlice+model_33/conv2d_transpose_55/Shape:output:0;model_33/conv2d_transpose_55/strided_slice_1/stack:output:0=model_33/conv2d_transpose_55/strided_slice_1/stack_1:output:0=model_33/conv2d_transpose_55/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
2model_33/conv2d_transpose_55/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_55/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_55/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,model_33/conv2d_transpose_55/strided_slice_2StridedSlice+model_33/conv2d_transpose_55/Shape:output:0;model_33/conv2d_transpose_55/strided_slice_2/stack:output:0=model_33/conv2d_transpose_55/strided_slice_2/stack_1:output:0=model_33/conv2d_transpose_55/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_33/conv2d_transpose_55/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
 model_33/conv2d_transpose_55/mulMul5model_33/conv2d_transpose_55/strided_slice_1:output:0+model_33/conv2d_transpose_55/mul/y:output:0*
T0*
_output_shapes
: f
$model_33/conv2d_transpose_55/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
"model_33/conv2d_transpose_55/mul_1Mul5model_33/conv2d_transpose_55/strided_slice_2:output:0-model_33/conv2d_transpose_55/mul_1/y:output:0*
T0*
_output_shapes
: f
$model_33/conv2d_transpose_55/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
"model_33/conv2d_transpose_55/stackPack3model_33/conv2d_transpose_55/strided_slice:output:0$model_33/conv2d_transpose_55/mul:z:0&model_33/conv2d_transpose_55/mul_1:z:0-model_33/conv2d_transpose_55/stack/3:output:0*
N*
T0*
_output_shapes
:|
2model_33/conv2d_transpose_55/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4model_33/conv2d_transpose_55/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model_33/conv2d_transpose_55/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,model_33/conv2d_transpose_55/strided_slice_3StridedSlice+model_33/conv2d_transpose_55/stack:output:0;model_33/conv2d_transpose_55/strided_slice_3/stack:output:0=model_33/conv2d_transpose_55/strided_slice_3/stack_1:output:0=model_33/conv2d_transpose_55/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<model_33/conv2d_transpose_55/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_33_conv2d_transpose_55_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
-model_33/conv2d_transpose_55/conv2d_transposeConv2DBackpropInput+model_33/conv2d_transpose_55/stack:output:0Dmodel_33/conv2d_transpose_55/conv2d_transpose/ReadVariableOp:value:0/model_33/conv2d_transpose_54/Relu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
3model_33/conv2d_transpose_55/BiasAdd/ReadVariableOpReadVariableOp<model_33_conv2d_transpose_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$model_33/conv2d_transpose_55/BiasAddBiasAdd6model_33/conv2d_transpose_55/conv2d_transpose:output:0;model_33/conv2d_transpose_55/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
(model_33/conv2d_81/Conv2D/ReadVariableOpReadVariableOp1model_33_conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_33/conv2d_81/Conv2DConv2D-model_33/conv2d_transpose_55/BiasAdd:output:00model_33/conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
)model_33/conv2d_81/BiasAdd/ReadVariableOpReadVariableOp2model_33_conv2d_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_33/conv2d_81/BiasAddBiasAdd"model_33/conv2d_81/Conv2D:output:01model_33/conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+????????????????????????????
model_33/conv2d_81/ReluRelu#model_33/conv2d_81/BiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentity%model_33/conv2d_81/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp*^model_33/conv2d_79/BiasAdd/ReadVariableOp)^model_33/conv2d_79/Conv2D/ReadVariableOp*^model_33/conv2d_80/BiasAdd/ReadVariableOp)^model_33/conv2d_80/Conv2D/ReadVariableOp*^model_33/conv2d_81/BiasAdd/ReadVariableOp)^model_33/conv2d_81/Conv2D/ReadVariableOp4^model_33/conv2d_transpose_54/BiasAdd/ReadVariableOp=^model_33/conv2d_transpose_54/conv2d_transpose/ReadVariableOp4^model_33/conv2d_transpose_55/BiasAdd/ReadVariableOp=^model_33/conv2d_transpose_55/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 2V
)model_33/conv2d_79/BiasAdd/ReadVariableOp)model_33/conv2d_79/BiasAdd/ReadVariableOp2T
(model_33/conv2d_79/Conv2D/ReadVariableOp(model_33/conv2d_79/Conv2D/ReadVariableOp2V
)model_33/conv2d_80/BiasAdd/ReadVariableOp)model_33/conv2d_80/BiasAdd/ReadVariableOp2T
(model_33/conv2d_80/Conv2D/ReadVariableOp(model_33/conv2d_80/Conv2D/ReadVariableOp2V
)model_33/conv2d_81/BiasAdd/ReadVariableOp)model_33/conv2d_81/BiasAdd/ReadVariableOp2T
(model_33/conv2d_81/Conv2D/ReadVariableOp(model_33/conv2d_81/Conv2D/ReadVariableOp2j
3model_33/conv2d_transpose_54/BiasAdd/ReadVariableOp3model_33/conv2d_transpose_54/BiasAdd/ReadVariableOp2|
<model_33/conv2d_transpose_54/conv2d_transpose/ReadVariableOp<model_33/conv2d_transpose_54/conv2d_transpose/ReadVariableOp2j
3model_33/conv2d_transpose_55/BiasAdd/ReadVariableOp3model_33/conv2d_transpose_55/BiasAdd/ReadVariableOp2|
<model_33/conv2d_transpose_55/conv2d_transpose/ReadVariableOp<model_33/conv2d_transpose_55/conv2d_transpose/ReadVariableOp:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
input_37
?
?
5__inference_conv2d_transpose_55_layer_call_fn_3914375

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_3913657?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_80_layer_call_and_return_conditional_losses_3914299

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????
j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3914319

inputs
identity?
MaxPoolMaxPoolinputs*A
_output_shapes/
-:+???????????????????????????
*
ksize
*
paddingSAME*
strides
r
IdentityIdentityMaxPool:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????
:i e
A
_output_shapes/
-:+???????????????????????????

 
_user_specified_nameinputs
?W
?
 __inference__traced_save_3914574
file_prefix/
+savev2_conv2d_79_kernel_read_readvariableop-
)savev2_conv2d_79_bias_read_readvariableop/
+savev2_conv2d_80_kernel_read_readvariableop-
)savev2_conv2d_80_bias_read_readvariableop9
5savev2_conv2d_transpose_54_kernel_read_readvariableop7
3savev2_conv2d_transpose_54_bias_read_readvariableop9
5savev2_conv2d_transpose_55_kernel_read_readvariableop7
3savev2_conv2d_transpose_55_bias_read_readvariableop/
+savev2_conv2d_81_kernel_read_readvariableop-
)savev2_conv2d_81_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_conv2d_79_kernel_m_read_readvariableop4
0savev2_adam_conv2d_79_bias_m_read_readvariableop6
2savev2_adam_conv2d_80_kernel_m_read_readvariableop4
0savev2_adam_conv2d_80_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_54_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_54_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_55_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_55_bias_m_read_readvariableop6
2savev2_adam_conv2d_81_kernel_m_read_readvariableop4
0savev2_adam_conv2d_81_bias_m_read_readvariableop6
2savev2_adam_conv2d_79_kernel_v_read_readvariableop4
0savev2_adam_conv2d_79_bias_v_read_readvariableop6
2savev2_adam_conv2d_80_kernel_v_read_readvariableop4
0savev2_adam_conv2d_80_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_54_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_54_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_55_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_55_bias_v_read_readvariableop6
2savev2_adam_conv2d_81_kernel_v_read_readvariableop4
0savev2_adam_conv2d_81_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_79_kernel_read_readvariableop)savev2_conv2d_79_bias_read_readvariableop+savev2_conv2d_80_kernel_read_readvariableop)savev2_conv2d_80_bias_read_readvariableop5savev2_conv2d_transpose_54_kernel_read_readvariableop3savev2_conv2d_transpose_54_bias_read_readvariableop5savev2_conv2d_transpose_55_kernel_read_readvariableop3savev2_conv2d_transpose_55_bias_read_readvariableop+savev2_conv2d_81_kernel_read_readvariableop)savev2_conv2d_81_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_conv2d_79_kernel_m_read_readvariableop0savev2_adam_conv2d_79_bias_m_read_readvariableop2savev2_adam_conv2d_80_kernel_m_read_readvariableop0savev2_adam_conv2d_80_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_54_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_54_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_55_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_55_bias_m_read_readvariableop2savev2_adam_conv2d_81_kernel_m_read_readvariableop0savev2_adam_conv2d_81_bias_m_read_readvariableop2savev2_adam_conv2d_79_kernel_v_read_readvariableop0savev2_adam_conv2d_79_bias_v_read_readvariableop2savev2_adam_conv2d_80_kernel_v_read_readvariableop0savev2_adam_conv2d_80_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_54_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_54_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_55_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_55_bias_v_read_readvariableop2savev2_adam_conv2d_81_kernel_v_read_readvariableop0savev2_adam_conv2d_81_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::
:
:
:::::: : : : : : : : : : : :::
:
:
::::::::
:
:
:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:
: #

_output_shapes
:
:,$(
&
_output_shapes
:
: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::*

_output_shapes
: 
?
?
+__inference_conv2d_81_layer_call_fn_3914417

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_3913738?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?#
?
P__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_3914366

inputsB
(conv2d_transpose_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:
*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????

 
_user_specified_nameinputs
?
?
F__inference_conv2d_81_layer_call_and_return_conditional_losses_3913738

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3913715

inputs
identity?
MaxPoolMaxPoolinputs*A
_output_shapes/
-:+???????????????????????????
*
ksize
*
paddingSAME*
strides
r
IdentityIdentityMaxPool:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????
:i e
A
_output_shapes/
-:+???????????????????????????

 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3914274

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_model_33_layer_call_fn_3914059

inputs!
unknown:
	unknown_0:#
	unknown_1:

	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_33_layer_call_and_return_conditional_losses_3913866?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_28_layer_call_fn_3914269

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3913692z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?&
?
E__inference_model_33_layer_call_and_return_conditional_losses_3913866

inputs+
conv2d_79_3913838:
conv2d_79_3913840:+
conv2d_80_3913844:

conv2d_80_3913846:
5
conv2d_transpose_54_3913850:
)
conv2d_transpose_54_3913852:5
conv2d_transpose_55_3913855:)
conv2d_transpose_55_3913857:+
conv2d_81_3913860:
conv2d_81_3913862:
identity??!conv2d_79/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall?+conv2d_transpose_54/StatefulPartitionedCall?+conv2d_transpose_55/StatefulPartitionedCall?
!conv2d_79/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_79_3913838conv2d_79_3913840*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_79_layer_call_and_return_conditional_losses_3913682?
 max_pooling2d_28/PartitionedCallPartitionedCall*conv2d_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3913692?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_28/PartitionedCall:output:0conv2d_80_3913844conv2d_80_3913846*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_3913705?
 max_pooling2d_29/PartitionedCallPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3913715?
+conv2d_transpose_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_29/PartitionedCall:output:0conv2d_transpose_54_3913850conv2d_transpose_54_3913852*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_3913613?
+conv2d_transpose_55/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_54/StatefulPartitionedCall:output:0conv2d_transpose_55_3913855conv2d_transpose_55_3913857*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_3913657?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_55/StatefulPartitionedCall:output:0conv2d_81_3913860conv2d_81_3913862*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_3913738?
IdentityIdentity*conv2d_81/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp"^conv2d_79/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall,^conv2d_transpose_54/StatefulPartitionedCall,^conv2d_transpose_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 2F
!conv2d_79/StatefulPartitionedCall!conv2d_79/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2Z
+conv2d_transpose_54/StatefulPartitionedCall+conv2d_transpose_54/StatefulPartitionedCall2Z
+conv2d_transpose_55/StatefulPartitionedCall+conv2d_transpose_55/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_54_layer_call_fn_3914328

inputs!
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_3913613?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????

 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3914314

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?&
?
E__inference_model_33_layer_call_and_return_conditional_losses_3913945
input_37+
conv2d_79_3913917:
conv2d_79_3913919:+
conv2d_80_3913923:

conv2d_80_3913925:
5
conv2d_transpose_54_3913929:
)
conv2d_transpose_54_3913931:5
conv2d_transpose_55_3913934:)
conv2d_transpose_55_3913936:+
conv2d_81_3913939:
conv2d_81_3913941:
identity??!conv2d_79/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall?+conv2d_transpose_54/StatefulPartitionedCall?+conv2d_transpose_55/StatefulPartitionedCall?
!conv2d_79/StatefulPartitionedCallStatefulPartitionedCallinput_37conv2d_79_3913917conv2d_79_3913919*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_79_layer_call_and_return_conditional_losses_3913682?
 max_pooling2d_28/PartitionedCallPartitionedCall*conv2d_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3913692?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_28/PartitionedCall:output:0conv2d_80_3913923conv2d_80_3913925*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_3913705?
 max_pooling2d_29/PartitionedCallPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3913715?
+conv2d_transpose_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_29/PartitionedCall:output:0conv2d_transpose_54_3913929conv2d_transpose_54_3913931*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_3913613?
+conv2d_transpose_55/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_54/StatefulPartitionedCall:output:0conv2d_transpose_55_3913934conv2d_transpose_55_3913936*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_3913657?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_55/StatefulPartitionedCall:output:0conv2d_81_3913939conv2d_81_3913941*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_3913738?
IdentityIdentity*conv2d_81/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp"^conv2d_79/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall,^conv2d_transpose_54/StatefulPartitionedCall,^conv2d_transpose_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 2F
!conv2d_79/StatefulPartitionedCall!conv2d_79/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2Z
+conv2d_transpose_54/StatefulPartitionedCall+conv2d_transpose_54/StatefulPartitionedCall2Z
+conv2d_transpose_55/StatefulPartitionedCall+conv2d_transpose_55/StatefulPartitionedCall:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
input_37
?
N
2__inference_max_pooling2d_29_layer_call_fn_3914304

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3913568?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_79_layer_call_fn_3914248

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_79_layer_call_and_return_conditional_losses_3913682?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3914279

inputs
identity?
MaxPoolMaxPoolinputs*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
r
IdentityIdentityMaxPool:output:0*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_29_layer_call_fn_3914309

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3913715z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????
:i e
A
_output_shapes/
-:+???????????????????????????

 
_user_specified_nameinputs
?
?
*__inference_model_33_layer_call_fn_3913768
input_37!
unknown:
	unknown_0:#
	unknown_1:

	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_37unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_33_layer_call_and_return_conditional_losses_3913745?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
input_37
?
?
*__inference_model_33_layer_call_fn_3914034

inputs!
unknown:
	unknown_0:#
	unknown_1:

	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_33_layer_call_and_return_conditional_losses_3913745?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3913692

inputs
identity?
MaxPoolMaxPoolinputs*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
r
IdentityIdentityMaxPool:output:0*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3913568

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_model_33_layer_call_fn_3913914
input_37!
unknown:
	unknown_0:#
	unknown_1:

	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_37unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_model_33_layer_call_and_return_conditional_losses_3913866?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
input_37
?&
?
E__inference_model_33_layer_call_and_return_conditional_losses_3913745

inputs+
conv2d_79_3913683:
conv2d_79_3913685:+
conv2d_80_3913706:

conv2d_80_3913708:
5
conv2d_transpose_54_3913717:
)
conv2d_transpose_54_3913719:5
conv2d_transpose_55_3913722:)
conv2d_transpose_55_3913724:+
conv2d_81_3913739:
conv2d_81_3913741:
identity??!conv2d_79/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall?+conv2d_transpose_54/StatefulPartitionedCall?+conv2d_transpose_55/StatefulPartitionedCall?
!conv2d_79/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_79_3913683conv2d_79_3913685*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_79_layer_call_and_return_conditional_losses_3913682?
 max_pooling2d_28/PartitionedCallPartitionedCall*conv2d_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3913692?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_28/PartitionedCall:output:0conv2d_80_3913706conv2d_80_3913708*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_3913705?
 max_pooling2d_29/PartitionedCallPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3913715?
+conv2d_transpose_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_29/PartitionedCall:output:0conv2d_transpose_54_3913717conv2d_transpose_54_3913719*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_3913613?
+conv2d_transpose_55/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_54/StatefulPartitionedCall:output:0conv2d_transpose_55_3913722conv2d_transpose_55_3913724*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_3913657?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_55/StatefulPartitionedCall:output:0conv2d_81_3913739conv2d_81_3913741*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_3913738?
IdentityIdentity*conv2d_81/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp"^conv2d_79/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall,^conv2d_transpose_54/StatefulPartitionedCall,^conv2d_transpose_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:+???????????????????????????: : : : : : : : : : 2F
!conv2d_79/StatefulPartitionedCall!conv2d_79/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2Z
+conv2d_transpose_54/StatefulPartitionedCall+conv2d_transpose_54/StatefulPartitionedCall2Z
+conv2d_transpose_55/StatefulPartitionedCall+conv2d_transpose_55/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
? 
?
P__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_3914408

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_81_layer_call_and_return_conditional_losses_3914428

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_28_layer_call_fn_3914264

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3913556?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
W
input_37K
serving_default_input_37:0+???????????????????????????W
	conv2d_81J
StatefulPartitionedCall:0+???????????????????????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
6iter

7beta_1

8beta_2
	9decay
:learning_ratemtmumvmw$mx%my*mz+m{0m|1m}v~vv?v?$v?%v?*v?+v?0v?1v?"
	optimizer
f
0
1
2
3
$4
%5
*6
+7
08
19"
trackable_list_wrapper
f
0
1
2
3
$4
%5
*6
+7
08
19"
trackable_list_wrapper
 "
trackable_list_wrapper
?

;layers

	variables
<non_trainable_variables
=layer_metrics
>layer_regularization_losses
?metrics
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
*:(2conv2d_79/kernel
:2conv2d_79/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

@layers
	variables
Anon_trainable_variables
Blayer_metrics
Clayer_regularization_losses
Dmetrics
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Elayers
	variables
Fnon_trainable_variables
Glayer_metrics
Hlayer_regularization_losses
Imetrics
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
2conv2d_80/kernel
:
2conv2d_80/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Jlayers
	variables
Knon_trainable_variables
Llayer_metrics
Mlayer_regularization_losses
Nmetrics
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Olayers
 	variables
Pnon_trainable_variables
Qlayer_metrics
Rlayer_regularization_losses
Smetrics
!trainable_variables
"regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2
2conv2d_transpose_54/kernel
&:$2conv2d_transpose_54/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Tlayers
&	variables
Unon_trainable_variables
Vlayer_metrics
Wlayer_regularization_losses
Xmetrics
'trainable_variables
(regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:22conv2d_transpose_55/kernel
&:$2conv2d_transpose_55/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Ylayers
,	variables
Znon_trainable_variables
[layer_metrics
\layer_regularization_losses
]metrics
-trainable_variables
.regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_81/kernel
:2conv2d_81/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?

^layers
2	variables
_non_trainable_variables
`layer_metrics
alayer_regularization_losses
bmetrics
3trainable_variables
4regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
c0
d1
e2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	ftotal
	gcount
h	variables
i	keras_api"
_tf_keras_metric
^
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api"
_tf_keras_metric
^
	ototal
	pcount
q
_fn_kwargs
r	variables
s	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
o0
p1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
/:-2Adam/conv2d_79/kernel/m
!:2Adam/conv2d_79/bias/m
/:-
2Adam/conv2d_80/kernel/m
!:
2Adam/conv2d_80/bias/m
9:7
2!Adam/conv2d_transpose_54/kernel/m
+:)2Adam/conv2d_transpose_54/bias/m
9:72!Adam/conv2d_transpose_55/kernel/m
+:)2Adam/conv2d_transpose_55/bias/m
/:-2Adam/conv2d_81/kernel/m
!:2Adam/conv2d_81/bias/m
/:-2Adam/conv2d_79/kernel/v
!:2Adam/conv2d_79/bias/v
/:-
2Adam/conv2d_80/kernel/v
!:
2Adam/conv2d_80/bias/v
9:7
2!Adam/conv2d_transpose_54/kernel/v
+:)2Adam/conv2d_transpose_54/bias/v
9:72!Adam/conv2d_transpose_55/kernel/v
+:)2Adam/conv2d_transpose_55/bias/v
/:-2Adam/conv2d_81/kernel/v
!:2Adam/conv2d_81/bias/v
?B?
"__inference__wrapped_model_3913547input_37"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_model_33_layer_call_fn_3913768
*__inference_model_33_layer_call_fn_3914034
*__inference_model_33_layer_call_fn_3914059
*__inference_model_33_layer_call_fn_3913914?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_model_33_layer_call_and_return_conditional_losses_3914149
E__inference_model_33_layer_call_and_return_conditional_losses_3914239
E__inference_model_33_layer_call_and_return_conditional_losses_3913945
E__inference_model_33_layer_call_and_return_conditional_losses_3913976?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_conv2d_79_layer_call_fn_3914248?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_79_layer_call_and_return_conditional_losses_3914259?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_max_pooling2d_28_layer_call_fn_3914264
2__inference_max_pooling2d_28_layer_call_fn_3914269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3914274
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3914279?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_80_layer_call_fn_3914288?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_80_layer_call_and_return_conditional_losses_3914299?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_max_pooling2d_29_layer_call_fn_3914304
2__inference_max_pooling2d_29_layer_call_fn_3914309?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3914314
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3914319?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv2d_transpose_54_layer_call_fn_3914328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_3914366?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv2d_transpose_55_layer_call_fn_3914375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_3914408?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_81_layer_call_fn_3914417?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_81_layer_call_and_return_conditional_losses_3914428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_3914009input_37"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_3913547?
$%*+01K?H
A?>
<?9
input_37+???????????????????????????
? "O?L
J
	conv2d_81=?:
	conv2d_81+????????????????????????????
F__inference_conv2d_79_layer_call_and_return_conditional_losses_3914259?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_conv2d_79_layer_call_fn_3914248?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_conv2d_80_layer_call_and_return_conditional_losses_3914299?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????

? ?
+__inference_conv2d_80_layer_call_fn_3914288?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????
?
F__inference_conv2d_81_layer_call_and_return_conditional_losses_3914428?01I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_conv2d_81_layer_call_fn_3914417?01I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
P__inference_conv2d_transpose_54_layer_call_and_return_conditional_losses_3914366?$%I?F
??<
:?7
inputs+???????????????????????????

? "??<
5?2
0+???????????????????????????
? ?
5__inference_conv2d_transpose_54_layer_call_fn_3914328?$%I?F
??<
:?7
inputs+???????????????????????????

? "2?/+????????????????????????????
P__inference_conv2d_transpose_55_layer_call_and_return_conditional_losses_3914408?*+I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
5__inference_conv2d_transpose_55_layer_call_fn_3914375?*+I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3914274?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_3914279?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
2__inference_max_pooling2d_28_layer_call_fn_3914264?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
2__inference_max_pooling2d_28_layer_call_fn_3914269I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3914314?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_3914319?I?F
??<
:?7
inputs+???????????????????????????

? "??<
5?2
0+???????????????????????????

? ?
2__inference_max_pooling2d_29_layer_call_fn_3914304?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
2__inference_max_pooling2d_29_layer_call_fn_3914309I?F
??<
:?7
inputs+???????????????????????????

? "2?/+???????????????????????????
?
E__inference_model_33_layer_call_and_return_conditional_losses_3913945?
$%*+01S?P
I?F
<?9
input_37+???????????????????????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_model_33_layer_call_and_return_conditional_losses_3913976?
$%*+01S?P
I?F
<?9
input_37+???????????????????????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_model_33_layer_call_and_return_conditional_losses_3914149?
$%*+01Q?N
G?D
:?7
inputs+???????????????????????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_model_33_layer_call_and_return_conditional_losses_3914239?
$%*+01Q?N
G?D
:?7
inputs+???????????????????????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
*__inference_model_33_layer_call_fn_3913768?
$%*+01S?P
I?F
<?9
input_37+???????????????????????????
p 

 
? "2?/+????????????????????????????
*__inference_model_33_layer_call_fn_3913914?
$%*+01S?P
I?F
<?9
input_37+???????????????????????????
p

 
? "2?/+????????????????????????????
*__inference_model_33_layer_call_fn_3914034?
$%*+01Q?N
G?D
:?7
inputs+???????????????????????????
p 

 
? "2?/+????????????????????????????
*__inference_model_33_layer_call_fn_3914059?
$%*+01Q?N
G?D
:?7
inputs+???????????????????????????
p

 
? "2?/+????????????????????????????
%__inference_signature_wrapper_3914009?
$%*+01W?T
? 
M?J
H
input_37<?9
input_37+???????????????????????????"O?L
J
	conv2d_81=?:
	conv2d_81+???????????????????????????