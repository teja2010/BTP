require 'torch'
require 'nn'
require 'math'
require 'optim'
require 'image'
require 'gnuplot'
require 'csvigo'

f = torch.load('old_con.t7','ascii')
img = f.images;
print(img:size())
lab1 = f.target1;
print(lab1:size())
lab2 = f.target2
print(lab2:size())

aa = lab2[{{1,10},{1}}];
print(aa)
file = io.open('one.txt','w');
io.output(file)

gnuplot.pngfigure('one.png')
gnuplot.plot({
	torch.range(1, aa:size(1)), -- x-coords = {1,2, ... ,#losses}
	torch.Tensor(aa),  -- y-coordinates (the training losses)
	'-'})
gnuplot.plotflush();


