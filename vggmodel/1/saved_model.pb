ż
ź
:
Add
x"T
y"T
z"T"
Ttype:
2	
E
AssignSubVariableOp
resource
value"dtype"
dtypetype
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:
2"
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

FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%ˇŃ8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ô
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
ö
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*1.14.02v1.14.0-rc1-22-gaf24dc9ä

input_tensorPlaceholder*
dtype0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
shape:˙˙˙˙˙˙˙˙˙
­
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_2/kernel*%
valueB"             

.conv2d_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *ž

.conv2d_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *>
ö
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 
Ú
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_2/kernel
ô
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*"
_class
loc:@conv2d_2/kernel
ć
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
: 
ˇ
conv2d_2/kernelVarHandleOp*
shape: *
dtype0*
_output_shapes
: * 
shared_nameconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
	container 
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 

conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_2/kernel*
dtype0

#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
: 

conv2d_2/bias/Initializer/zerosConst* 
_class
loc:@conv2d_2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Ľ
conv2d_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
	container *
shape: 
k
.conv2d_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 

conv2d_2/bias/AssignAssignVariableOpconv2d_2/biasconv2d_2/bias/Initializer/zeros* 
_class
loc:@conv2d_2/bias*
dtype0

!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 
g
conv2d_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
: 

conv2d_2/Conv2DConv2Dinput_tensorconv2d_2/Conv2D/ReadVariableOp*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
i
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
: 

conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
a
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
­
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_3/kernel*%
valueB"              *
dtype0*
_output_shapes
:

.conv2d_3/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *ěŃ˝*
dtype0*
_output_shapes
: 

.conv2d_3/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel*
valueB
 *ěŃ=
ö
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:  *

seed *
T0*"
_class
loc:@conv2d_3/kernel*
seed2 
Ú
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
ô
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:  
ć
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:  
ˇ
conv2d_3/kernelVarHandleOp*
shape:  *
dtype0*
_output_shapes
: * 
shared_nameconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
	container 
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 

conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_3/kernel*
dtype0

#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*&
_output_shapes
:  

conv2d_3/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_3/bias*
valueB *    
Ľ
conv2d_3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
	container *
shape: 
k
.conv2d_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/bias*
_output_shapes
: 

conv2d_3/bias/AssignAssignVariableOpconv2d_3/biasconv2d_3/bias/Initializer/zeros*
dtype0* 
_class
loc:@conv2d_3/bias

!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
: 
g
conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*&
_output_shapes
:  

conv2d_3/Conv2DConv2Dconv2d_2/Reluconv2d_3/Conv2D/ReadVariableOp*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
i
conv2d_3/BiasAdd/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes
: 

conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/BiasAdd/ReadVariableOp*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
a
conv2d_3/ReluReluconv2d_3/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ž
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/Relu*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
\
keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
n
dropout_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
]
dropout_1/cond/switch_tIdentitydropout_1/cond/Switch:1*
T0
*
_output_shapes
: 
[
dropout_1/cond/switch_fIdentitydropout_1/cond/Switch*
T0
*
_output_shapes
: 
Y
dropout_1/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

z
dropout_1/cond/dropout/rateConst^dropout_1/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 

dropout_1/cond/dropout/ShapeShape%dropout_1/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
ß
#dropout_1/cond/dropout/Shape/SwitchSwitchmax_pooling2d_1/MaxPooldropout_1/cond/pred_id*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 

)dropout_1/cond/dropout/random_uniform/minConst^dropout_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

)dropout_1/cond/dropout/random_uniform/maxConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Â
3dropout_1/cond/dropout/random_uniform/RandomUniformRandomUniformdropout_1/cond/dropout/Shape*
T0*
dtype0*
seed2 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *

seed 
§
)dropout_1/cond/dropout/random_uniform/subSub)dropout_1/cond/dropout/random_uniform/max)dropout_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
Ę
)dropout_1/cond/dropout/random_uniform/mulMul3dropout_1/cond/dropout/random_uniform/RandomUniform)dropout_1/cond/dropout/random_uniform/sub*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ź
%dropout_1/cond/dropout/random_uniformAdd)dropout_1/cond/dropout/random_uniform/mul)dropout_1/cond/dropout/random_uniform/min*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
{
dropout_1/cond/dropout/sub/xConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
dropout_1/cond/dropout/subSubdropout_1/cond/dropout/sub/xdropout_1/cond/dropout/rate*
T0*
_output_shapes
: 

 dropout_1/cond/dropout/truediv/xConst^dropout_1/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 

dropout_1/cond/dropout/truedivRealDiv dropout_1/cond/dropout/truediv/xdropout_1/cond/dropout/sub*
_output_shapes
: *
T0
ą
#dropout_1/cond/dropout/GreaterEqualGreaterEqual%dropout_1/cond/dropout/random_uniformdropout_1/cond/dropout/rate*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
˘
dropout_1/cond/dropout/mulMul%dropout_1/cond/dropout/Shape/Switch:1dropout_1/cond/dropout/truediv*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ą
dropout_1/cond/dropout/CastCast#dropout_1/cond/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

dropout_1/cond/dropout/mul_1Muldropout_1/cond/dropout/muldropout_1/cond/dropout/Cast*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
}
dropout_1/cond/IdentityIdentitydropout_1/cond/Identity/Switch*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ú
dropout_1/cond/Identity/SwitchSwitchmax_pooling2d_1/MaxPooldropout_1/cond/pred_id*
T0**
_class 
loc:@max_pooling2d_1/MaxPool*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 

dropout_1/cond/MergeMergedropout_1/cond/Identitydropout_1/cond/dropout/mul_1*
T0*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ : 
Š
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
valueB *  ?*
dtype0*
_output_shapes
: 
Ď
batch_normalization_1/gammaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape: 

<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
ž
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0
ˇ
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_1/gamma
¨
,batch_normalization_1/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
valueB *    *
dtype0*
_output_shapes
: 
Ě
batch_normalization_1/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape: 

;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
ť
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_1/beta*
dtype0
´
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
: 
ś
3batch_normalization_1/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB *    *
dtype0*
_output_shapes
: 
á
!batch_normalization_1/moving_meanVarHandleOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_1/moving_mean

Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
×
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
É
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
˝
6batch_normalization_1/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB *  ?*
dtype0*
_output_shapes
: 
í
%batch_normalization_1/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: 

Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
ć
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
Ő
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: 
z
!batch_normalization_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_1/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 

)batch_normalization_1/cond/ReadVariableOpReadVariableOp2batch_normalization_1/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
: 
Î
0batch_normalization_1/cond/ReadVariableOp/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 

+batch_normalization_1/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_1/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
: 
Î
2batch_normalization_1/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 

 batch_normalization_1/cond/ConstConst$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_1/cond/Const_1Const$^batch_normalization_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 

)batch_normalization_1/cond/FusedBatchNormFusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm/Switch:1)batch_normalization_1/cond/ReadVariableOp+batch_normalization_1/cond/ReadVariableOp_1 batch_normalization_1/cond/Const"batch_normalization_1/cond/Const_1*
T0*
data_formatNHWC*
is_training(*G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙ : : : : *
epsilon%o:
ň
0batch_normalization_1/cond/FusedBatchNorm/SwitchSwitchdropout_1/cond/Merge"batch_normalization_1/cond/pred_id*
T0*'
_class
loc:@dropout_1/cond/Merge*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 

+batch_normalization_1/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
: 
Đ
2batch_normalization_1/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 

+batch_normalization_1/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
: 
Î
2batch_normalization_1/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
¸
:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
: 
ë
Abatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_1/moving_mean"batch_normalization_1/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: : 
ź
<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
: 
ő
Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_1/moving_variance"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: : 
Ó
+batch_normalization_1/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm_1/Switch+batch_normalization_1/cond/ReadVariableOp_2+batch_normalization_1/cond/ReadVariableOp_3:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1*
data_formatNHWC*
is_training( *G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙ : : : : *
epsilon%o:*
T0
ô
2batch_normalization_1/cond/FusedBatchNorm_1/SwitchSwitchdropout_1/cond/Merge"batch_normalization_1/cond/pred_id*
T0*'
_class
loc:@dropout_1/cond/Merge*J
_output_shapes8
6:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
Ć
 batch_normalization_1/cond/MergeMerge+batch_normalization_1/cond/FusedBatchNorm_1)batch_normalization_1/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ : 
ˇ
"batch_normalization_1/cond/Merge_1Merge-batch_normalization_1/cond/FusedBatchNorm_1:1+batch_normalization_1/cond/FusedBatchNorm:1*
N*
_output_shapes

: : *
T0
ˇ
"batch_normalization_1/cond/Merge_2Merge-batch_normalization_1/cond/FusedBatchNorm_1:2+batch_normalization_1/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

: : 
|
#batch_normalization_1/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_1/cond_1/switch_tIdentity%batch_normalization_1/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_1/cond_1/switch_fIdentity#batch_normalization_1/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_1/cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0


"batch_normalization_1/cond_1/ConstConst&^batch_normalization_1/cond_1/switch_t*
valueB
 *¤p}?*
dtype0*
_output_shapes
: 

$batch_normalization_1/cond_1/Const_1Const&^batch_normalization_1/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ą
"batch_normalization_1/cond_1/MergeMerge$batch_normalization_1/cond_1/Const_1"batch_normalization_1/cond_1/Const*
T0*
N*
_output_shapes
: : 
Ś
+batch_normalization_1/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ř
)batch_normalization_1/AssignMovingAvg/subSub+batch_normalization_1/AssignMovingAvg/sub/x"batch_normalization_1/cond_1/Merge*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean

4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
ç
+batch_normalization_1/AssignMovingAvg/sub_1Sub4batch_normalization_1/AssignMovingAvg/ReadVariableOp"batch_normalization_1/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
ă
)batch_normalization_1/AssignMovingAvg/mulMul+batch_normalization_1/AssignMovingAvg/sub_1)batch_normalization_1/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
á
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*
dtype0*4
_class*
(&loc:@batch_normalization_1/moving_mean

6batch_normalization_1/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
Ź
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ŕ
+batch_normalization_1/AssignMovingAvg_1/subSub-batch_normalization_1/AssignMovingAvg_1/sub/x"batch_normalization_1/cond_1/Merge*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance

6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: 
ď
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp"batch_normalization_1/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
í
+batch_normalization_1/AssignMovingAvg_1/mulMul-batch_normalization_1/AssignMovingAvg_1/sub_1+batch_normalization_1/AssignMovingAvg_1/sub*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
í
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0

8batch_normalization_1/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_1/moving_variance
m
flatten/ShapeShape batch_normalization_1/cond/Merge*
T0*
out_type0*
_output_shapes
:
e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ą
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
b
flatten/Reshape/shape/1Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:

flatten/ReshapeReshape batch_normalization_1/cond/Mergeflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙1

-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"     *
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *çÓúź*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *çÓú<*
dtype0*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
1*

seed *
T0*
_class
loc:@dense/kernel*
seed2 
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
1
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min* 
_output_shapes
:
1*
T0*
_class
loc:@dense/kernel
¨
dense/kernelVarHandleOp*
	container *
shape:
1*
dtype0*
_output_shapes
: *
shared_namedense/kernel*
_class
loc:@dense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 

dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
dtype0

 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@dense/kernel*
dtype0* 
_output_shapes
:
1

dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@dense/bias*
valueB*    


dense/biasVarHandleOp*
shared_name
dense/bias*
_class
loc:@dense/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0*
_class
loc:@dense/bias

dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:
j
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0* 
_output_shapes
:
1

dense/MatMulMatMulflatten/Reshapedense/MatMul/ReadVariableOp*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 
d
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:

dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
T

dense/ReluReludense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
valueB"   
   *
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
valueB
 *ÍUž

-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
valueB
 *ÍU>
ě
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0*
_output_shapes
:	
*

seed 
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
é
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Ű
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

­
dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
	container *
shape:	
*
dtype0*
_output_shapes
: 
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 

dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
dtype0

"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	
*!
_class
loc:@dense_1/kernel

dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB
*    *
dtype0*
_output_shapes
:

˘
dense_1/biasVarHandleOp*
_class
loc:@dense_1/bias*
	container *
shape:
*
dtype0*
_output_shapes
: *
shared_namedense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 

dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
_class
loc:@dense_1/bias*
dtype0

 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

m
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	


dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_b( 
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:


dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

]
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

K
ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:*K
valueBB@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
c
RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
|
	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
B
IdentityIdentity	RestoreV2*
_output_shapes
:*
T0
L
AssignVariableOpAssignVariableOpconv2d_2/kernelIdentity*
dtype0

RestoreV2_1/tensor_namesConst*I
value@B>B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_1	RestoreV2ConstRestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_1IdentityRestoreV2_1*
T0*
_output_shapes
:
N
AssignVariableOp_1AssignVariableOpconv2d_2/bias
Identity_1*
dtype0

RestoreV2_2/tensor_namesConst*K
valueBB@B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_2	RestoreV2ConstRestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
F

Identity_2IdentityRestoreV2_2*
T0*
_output_shapes
:
P
AssignVariableOp_2AssignVariableOpconv2d_3/kernel
Identity_2*
dtype0

RestoreV2_3/tensor_namesConst*I
value@B>B4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_3/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_3	RestoreV2ConstRestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_3IdentityRestoreV2_3*
T0*
_output_shapes
:
N
AssignVariableOp_3AssignVariableOpconv2d_3/bias
Identity_3*
dtype0

RestoreV2_4/tensor_namesConst*J
valueAB?B5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_4	RestoreV2ConstRestoreV2_4/tensor_namesRestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_4IdentityRestoreV2_4*
T0*
_output_shapes
:
\
AssignVariableOp_4AssignVariableOpbatch_normalization_1/gamma
Identity_4*
dtype0

RestoreV2_5/tensor_namesConst*I
value@B>B4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_5	RestoreV2ConstRestoreV2_5/tensor_namesRestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_5IdentityRestoreV2_5*
T0*
_output_shapes
:
[
AssignVariableOp_5AssignVariableOpbatch_normalization_1/beta
Identity_5*
dtype0

RestoreV2_6/tensor_namesConst*P
valueGBEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_6/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_6	RestoreV2ConstRestoreV2_6/tensor_namesRestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_6IdentityRestoreV2_6*
T0*
_output_shapes
:
b
AssignVariableOp_6AssignVariableOp!batch_normalization_1/moving_mean
Identity_6*
dtype0
 
RestoreV2_7/tensor_namesConst*T
valueKBIB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_7/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_7	RestoreV2ConstRestoreV2_7/tensor_namesRestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_7IdentityRestoreV2_7*
_output_shapes
:*
T0
f
AssignVariableOp_7AssignVariableOp%batch_normalization_1/moving_variance
Identity_7*
dtype0

RestoreV2_8/tensor_namesConst*K
valueBB@B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_8/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_8	RestoreV2ConstRestoreV2_8/tensor_namesRestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_8IdentityRestoreV2_8*
_output_shapes
:*
T0
M
AssignVariableOp_8AssignVariableOpdense/kernel
Identity_8*
dtype0

RestoreV2_9/tensor_namesConst*I
value@B>B4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
e
RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_9	RestoreV2ConstRestoreV2_9/tensor_namesRestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
F

Identity_9IdentityRestoreV2_9*
_output_shapes
:*
T0
K
AssignVariableOp_9AssignVariableOp
dense/bias
Identity_9*
dtype0

RestoreV2_10/tensor_namesConst*
dtype0*
_output_shapes
:*K
valueBB@B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
f
RestoreV2_10/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_10	RestoreV2ConstRestoreV2_10/tensor_namesRestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
H
Identity_10IdentityRestoreV2_10*
T0*
_output_shapes
:
Q
AssignVariableOp_10AssignVariableOpdense_1/kernelIdentity_10*
dtype0

RestoreV2_11/tensor_namesConst*
dtype0*
_output_shapes
:*I
value@B>B4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
f
RestoreV2_11/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_11	RestoreV2ConstRestoreV2_11/tensor_namesRestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
H
Identity_11IdentityRestoreV2_11*
T0*
_output_shapes
:
O
AssignVariableOp_11AssignVariableOpdense_1/biasIdentity_11*
dtype0
g
VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
Q
VarIsInitializedOp_1VarIsInitializedOpconv2d_3/bias*
_output_shapes
: 
S
VarIsInitializedOp_2VarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
P
VarIsInitializedOp_3VarIsInitializedOpdense_1/bias*
_output_shapes
: 
N
VarIsInitializedOp_4VarIsInitializedOp
dense/bias*
_output_shapes
: 
Q
VarIsInitializedOp_5VarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
P
VarIsInitializedOp_6VarIsInitializedOpdense/kernel*
_output_shapes
: 
_
VarIsInitializedOp_7VarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
R
VarIsInitializedOp_8VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
^
VarIsInitializedOp_9VarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
f
VarIsInitializedOp_10VarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
T
VarIsInitializedOp_11VarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
ç
initNoOp"^batch_normalization_1/beta/Assign#^batch_normalization_1/gamma/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_8f61a73045564744996b2751a1d97a5c/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ý
save/SaveV2/tensor_namesConst*
valueBBbatch_normalization_1/betaBbatch_normalization_1/gammaB!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernel*
dtype0*
_output_shapes
:
{
save/SaveV2/shape_and_slicesConst*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ę
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices.batch_normalization_1/beta/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
ŕ
save/RestoreV2/tensor_namesConst*
valueBBbatch_normalization_1/betaBbatch_normalization_1/gammaB!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernel*
dtype0*
_output_shapes
:
~
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*+
value"B B B B B B B B B B B B B 
Ç
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*D
_output_shapes2
0::::::::::::*
dtypes
2
N
save/Identity_1Identitysave/RestoreV2*
_output_shapes
:*
T0
c
save/AssignVariableOpAssignVariableOpbatch_normalization_1/betasave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
f
save/AssignVariableOp_1AssignVariableOpbatch_normalization_1/gammasave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
_output_shapes
:*
T0
l
save/AssignVariableOp_2AssignVariableOp!batch_normalization_1/moving_meansave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
p
save/AssignVariableOp_3AssignVariableOp%batch_normalization_1/moving_variancesave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
X
save/AssignVariableOp_4AssignVariableOpconv2d_2/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
Z
save/AssignVariableOp_5AssignVariableOpconv2d_2/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
X
save/AssignVariableOp_6AssignVariableOpconv2d_3/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
Z
save/AssignVariableOp_7AssignVariableOpconv2d_3/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
U
save/AssignVariableOp_8AssignVariableOp
dense/biassave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
X
save/AssignVariableOp_9AssignVariableOpdense/kernelsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
Y
save/AssignVariableOp_10AssignVariableOpdense_1/biassave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
[
save/AssignVariableOp_11AssignVariableOpdense_1/kernelsave/Identity_12*
dtype0
Ň
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard "&<
save/Const:0save/Identity:0save/restore_all (5 @F8"Á

trainable_variablesŠ
Ś


conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08

conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
Ş
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
§
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"¸%
cond_context§%¤%
 
dropout_1/cond/cond_textdropout_1/cond/pred_id:0dropout_1/cond/switch_t:0 *Ě
dropout_1/cond/dropout/Cast:0
%dropout_1/cond/dropout/GreaterEqual:0
%dropout_1/cond/dropout/Shape/Switch:1
dropout_1/cond/dropout/Shape:0
dropout_1/cond/dropout/mul:0
dropout_1/cond/dropout/mul_1:0
5dropout_1/cond/dropout/random_uniform/RandomUniform:0
+dropout_1/cond/dropout/random_uniform/max:0
+dropout_1/cond/dropout/random_uniform/min:0
+dropout_1/cond/dropout/random_uniform/mul:0
+dropout_1/cond/dropout/random_uniform/sub:0
'dropout_1/cond/dropout/random_uniform:0
dropout_1/cond/dropout/rate:0
dropout_1/cond/dropout/sub/x:0
dropout_1/cond/dropout/sub:0
"dropout_1/cond/dropout/truediv/x:0
 dropout_1/cond/dropout/truediv:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_t:0
max_pooling2d_1/MaxPool:0B
max_pooling2d_1/MaxPool:0%dropout_1/cond/dropout/Shape/Switch:14
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0
Ö
dropout_1/cond/cond_text_1dropout_1/cond/pred_id:0dropout_1/cond/switch_f:0*
 dropout_1/cond/Identity/Switch:0
dropout_1/cond/Identity:0
dropout_1/cond/pred_id:0
dropout_1/cond/switch_f:0
max_pooling2d_1/MaxPool:0=
max_pooling2d_1/MaxPool:0 dropout_1/cond/Identity/Switch:04
dropout_1/cond/pred_id:0dropout_1/cond/pred_id:0
	
$batch_normalization_1/cond/cond_text$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_t:0 *
batch_normalization_1/beta:0
"batch_normalization_1/cond/Const:0
$batch_normalization_1/cond/Const_1:0
2batch_normalization_1/cond/FusedBatchNorm/Switch:1
+batch_normalization_1/cond/FusedBatchNorm:0
+batch_normalization_1/cond/FusedBatchNorm:1
+batch_normalization_1/cond/FusedBatchNorm:2
+batch_normalization_1/cond/FusedBatchNorm:3
+batch_normalization_1/cond/FusedBatchNorm:4
2batch_normalization_1/cond/ReadVariableOp/Switch:1
+batch_normalization_1/cond/ReadVariableOp:0
4batch_normalization_1/cond/ReadVariableOp_1/Switch:1
-batch_normalization_1/cond/ReadVariableOp_1:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_t:0
batch_normalization_1/gamma:0
dropout_1/cond/Merge:0T
batch_normalization_1/beta:04batch_normalization_1/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0S
batch_normalization_1/gamma:02batch_normalization_1/cond/ReadVariableOp/Switch:1L
dropout_1/cond/Merge:02batch_normalization_1/cond/FusedBatchNorm/Switch:1

&batch_normalization_1/cond/cond_text_1$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_f:0*
batch_normalization_1/beta:0
Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_1/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_1/cond/FusedBatchNorm_1:0
-batch_normalization_1/cond/FusedBatchNorm_1:1
-batch_normalization_1/cond/FusedBatchNorm_1:2
-batch_normalization_1/cond/FusedBatchNorm_1:3
-batch_normalization_1/cond/FusedBatchNorm_1:4
4batch_normalization_1/cond/ReadVariableOp_2/Switch:0
-batch_normalization_1/cond/ReadVariableOp_2:0
4batch_normalization_1/cond/ReadVariableOp_3/Switch:0
-batch_normalization_1/cond/ReadVariableOp_3:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_f:0
batch_normalization_1/gamma:0
#batch_normalization_1/moving_mean:0
'batch_normalization_1/moving_variance:0
dropout_1/cond/Merge:0U
batch_normalization_1/gamma:04batch_normalization_1/cond/ReadVariableOp_2/Switch:0N
dropout_1/cond/Merge:04batch_normalization_1/cond/FusedBatchNorm_1/Switch:0L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0j
#batch_normalization_1/moving_mean:0Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0p
'batch_normalization_1/moving_variance:0Ebatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0T
batch_normalization_1/beta:04batch_normalization_1/cond/ReadVariableOp_3/Switch:0
Ç
&batch_normalization_1/cond_1/cond_text&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_t:0 *É
$batch_normalization_1/cond_1/Const:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_t:0P
&batch_normalization_1/cond_1/pred_id:0&batch_normalization_1/cond_1/pred_id:0
É
(batch_normalization_1/cond_1/cond_text_1&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_f:0*Ë
&batch_normalization_1/cond_1/Const_1:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_f:0P
&batch_normalization_1/cond_1/pred_id:0&batch_normalization_1/cond_1/pred_id:0"Ö
	variablesČĹ

conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08

conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
Ş
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
§
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
Ĺ
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(25batch_normalization_1/moving_mean/Initializer/zeros:0@H
Ô
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(28batch_normalization_1/moving_variance/Initializer/ones:0@H
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08*Ž
serving_default
=
input_images-
input_tensor:0˙˙˙˙˙˙˙˙˙=
dense_1/Softmax:0(
dense_1/Softmax:0˙˙˙˙˙˙˙˙˙
tensorflow/serving/predict