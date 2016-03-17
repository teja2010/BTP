--0 libs
require 'torch'
require 'nn'
require 'math'
require 'optim'
require 'gnuplot'
--require 'cunn'
torch.manualSeed(123);

--********************
--1 data
part = 1000
tepart = 30
numClass=16
printter=1;
-- add validation tests ??

print('loading train data, size:')
f = torch.load('old_vow.t7','ascii')
imgs1 = f.images:type(torch.getdefaulttensortype());
labs1 = f.target;
--labs1 = torch.ones(imgs1:size(1));
f = nil;
print(imgs1:size())

--f = torch.load('old_con.t7','ascii')
--imgs2 = f.images:type(torch.getdefaulttensortype())
----
--labs2 = f.target1;
--labs2 = torch.ones(imgs2:size(1));
--labs2:add(1);
--f = nil;
--print(imgs2:size())
--
--img = torch.cat(imgs1,imgs2,1);
--lab = torch.cat(labs1,labs2,1);
img = imgs1;
lab = labs1;

print(img:size())
print(lab:size())
--randomze
function randomTT(im,la)
	local sz1 = im:size(1);
	local sz2 = im:size(2);
	local shuffle = torch.randperm(sz1);
	local shI = torch.Tensor(sz1,sz2);
	local shL = torch.Tensor(sz1);
	for i=1,sz1 do
		shI[{{i},{}}] = im[{{shuffle[i]},{}}];
		shL[{{i}}] = la[{{shuffle[i]}}];
	end

	return shI,shL;
end
imgs, labs = randomTT(img,lab);
--gnuplot.plot({
--	torch.range(1, labs:size(1)), -- x-coords = {1,2, ... ,#losses}
--	torch.Tensor(labs),  -- y-coordinates (the training losses)
--	'-'})
--print(labs[{{101,105},{1}}]);
--print(s2[{{101,105}}]);
--remove unused
img = nil; lab=nil;
imgs1 = nil;-- imgs2 =nil;
labs1 = nil;-- labs2 = nil; 

	--print(type(trD))
	--print(labs)
--
--train data and labels
trD = imgs[{{1,part},{}}]:resize(part,28,28);
trL = labs[{{1,part}}]:resize(part);
--trL = torch.LongTensor():resize(trL:size()):copy(trL)

print(trD:size())
print(trL:size())

--gnuplot.plot({
--	torch.range(1, trL:size(1)), -- x-coords = {1,2, ... ,#losses}
--	torch.Tensor(trL),  -- y-coordinates (the training losses)
--	'-'})

--print(type(trL))
--print(labs[{{1,10},{}}])
--trT = torch.zeros(part,16);
--for i=1,part do
--	--print(trL[i])
--	local cc = trL[i]
--	trT[{ {i},{cc} }]=1
--end
--trT:scatter(2,trL,trL)
--print(trT[{{1},{}}]);print(trT[{{2},{}}]);print(trT[{{10},{}}]);


--test data and labels
print('loading test data, size:')
teD = imgs[{{part+1,part+tepart},{}}]:resize(tepart,28,28);
teL = labs[{{part+1,part+tepart}}]:resize(tepart);
--teD = imgs[{{part+1,imgs:size(1)},{}}]:resize(imgs:size(1)-part,28,28);
--teL = labs[{{part+1,imgs:size(1)}}]:resize(imgs:size(1)-part);
print(teD:size())
print(teL:size())
--print(teL)
teL = torch.LongTensor():resize(teL:size()):copy(teL)
--teT = torch.zeros(teL:size(1),16);
----for i= 1,
--for i=1,teL:size(1) do
--	local cc = trL[i]
--	teT[{ {i},{cc} }]=1
--end
--print(teT)

--********************
--2 model
nin = 28*28;
print('num of inputs =' .. nin);

net= nn.Sequential()
net:add(nn.SpatialConvolutionMM(1, 50, 5, 5))
--net:add(nn.SpatialBatchNormalization(40,1e-2))
--net:add(nn.ReLU(true))
net:add(nn.Tanh())
net:add(nn.SpatialMaxPooling(3, 3, 3, 3))
net:add(nn.SpatialConvolutionMM(50, 100, 5, 5))
--net:add(nn.SpatialBatchNormalization(40,1e-2))
--net:add(nn.ReLU(true))
net:add(nn.Tanh())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.Reshape(100*2*2))

net:add(nn.Linear(100*2*2, 100))
net:add(nn.Tanh())
net:add(nn.Linear(100, 16))

net:add(nn.LogSoftMax())

logger = io.open('ll/vowel.txt','a')
print(net)
logger:write('\n#############################################\n\n')
logger:write(tostring(net))

--********************
--3 loss function
--negative log likelihood
criterion = nn.ClassNLLCriterion()
--criterion = nn.MSECriterion()

--4 train

parm,grad = net:getParameters()

count=0
batch=20

feval = function(p_n)
	
	if parm~=p_n then
		parm:copy(p_n)
	end
	stI = count*batch + 1;
	enI = math.min(part, (count+1)*batch);
	if enI == part then
		count =0;
	else
		count= count +1;
	end

	--randomization ??

	bIn = trD[{{stI,enI},{},{}}]:view(batch,1,28,28);
	bLab = trL[{{stI,enI}}];--:view(batch,1);
	--bLab = trT[{ {stI,enI}, {} }]-- for cunn
	
	--print(bLab);

	grad:zero()
	bOut = net:forward(bIn);
	bLoss = criterion:forward(bOut,bLab)
	dl_do = criterion:backward(bOut,bLab);
	net:backward(bIn,dl_do)

	return bLoss, grad

end

-- optim meth
losses = {}
epochs =5
iter = epochs * math.ceil(part/batch)
print('iter ='..iter);

optimState = {
	learningRate = 1e-1,
	weightDecay = 0,
	momentum = 0.8,
	learningRateDecay = 1e-5
}
optimMeth = optim.sgd

--optimState = {
--	maxIter =2,
--	lineSearch = optim.lswolfe
--	}
--optimMeth = optim.lbfgs;

for i=1,iter do
	_, miniBLoss = optimMeth(feval,parm,optimState)
	if i % 10 == 0 then
		print(string.format("minibatches %6s, loss = %6.6f",i,miniBLoss[1]))
	end
	losses[#losses + 1] = miniBLoss[1]
end

--plot the losses
gnuplot.pngfigure('log/conv_vowel.png')
gnuplot.plot({
	torch.range(1, #losses), -- x-coords = {1,2, ... ,#losses}
	torch.Tensor(losses),  -- y-coordinates (the training losses)
	'-'})
gnuplot.plotflush();


--train data output
testD = trD:view(part,1,28,28)--nin)
logProb = net:forward(testD[{}])
--logProb = net:forward(teD:view(teD:size(1),nin))
print(logProb:size())
classProb = torch.exp(logProb)
_, pred = torch.max(classProb,2)

corr=0
for i=1,part do
	--print(pred[1][1])
	--print(teL[1])
	if pred[i][1]== trL[i] then
		corr=corr+1;
	end
end

print('train preds: '.. corr*100/part)
logger:write('\n'..'train preds: '.. corr*100/part .. "\n")

if printter == 1 then
	logFileName1 = 'log/output_vowels_train.txt';
	fileout = io.open(logFileName1,'w');                                
	fileout:write("trL \t pred\n");                                                  
	for i =1,part do 
		fileout:write(tostring(trL[i]) .. "\t" .. tostring(pred[i][1]) .. "\n");
	end
	fileout:close()     
end

--test data output
testON = tepart
testD = teD[{{1,testON},{},{}}]:view(testON,1,28,28)--nin)
logProb = net:forward(testD[{}])
--logProb = net:forward(teD:view(teD:size(1),nin))
print(logProb:size())
classProb = torch.exp(logProb)
_, pred = torch.max(classProb,2)

corr=0
for i=1,testON do
	--print(pred[1][1])
	--print(teL[1])
	if pred[i][1]== teL[i] then
		corr=corr+1;
	end
end

print('test preds: '.. corr*100/testON)
logger:write('test preds: '.. corr*100/tepart .. "\n")
logger:write('\n*********************************************')
logger:close()
if printter == 1 then
	logFileName2 = 'log/output_vowels_test.txt';
	fileout = io.open(logFileName2,'w');                                
	fileout:write("teL \t pred\n");
	for i =1,testON do 
		fileout:write(tostring(teL[i]) .. "\t" .. tostring(pred[i][1]) .. "\n");
	end   
	fileout:close()     

	dofile 'confusion.lua'
end

--[[
]]--


