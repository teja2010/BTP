require 'torch';

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

--train
ff = io.open(logFileName1,'r');
io.input(ff);
io.read();
conf = torch.zeros(numClass,numClass)
while true do
	local line = io.read();
	if line == nil then break end
	local spl = line:split('\t');
	local a  = tonumber(spl[1]);
	local b  = tonumber(spl[2]);
	conf[a][b] = conf[a][b] + 1;
	--print(a .. b .. '\n')
end
ff:close()

ff = io.open((logFileName1 .. '_conf.txt'),'w');
io.output(ff);
for i = 1,numClass do
	local ss = '';
	ss = (ss .. conf[i][1])
	for j =2,numClass do
		ss = (ss .. "," .. conf[i][j])
		--print(conf[i][j] .. "\t");
	end
	ss = (ss .. "\n")
	--print("\n")
	io.write(ss)
end
ff:close()

--test
ff = io.open(logFileName2,'r');
io.input(ff);
io.read();
conf = torch.zeros(numClass,numClass)
while true do
	local line = io.read();
	if line == nil then break end
	local spl = line:split('\t');
	local a  = tonumber(spl[1]);
	local b  = tonumber(spl[2]);
	conf[a][b] = conf[a][b] + 1;
	--print(a .. b .. '\n')
end
ff:close()

ff = io.open((logFileName2 ..'_conf.txt'),'w');
io.output(ff);
for i = 1,numClass do
	local ss = '';
	ss = (ss .. conf[i][1])
	for j =2,numClass do
		ss = (ss .. "," .. conf[i][j])
		--print(conf[i][j] .. "\t");
	end
	ss = (ss .. "\n")
	--print("\n")
	io.write(ss)
end
ff:close()
