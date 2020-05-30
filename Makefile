SHELL:=bash
method:=Threshold
seq:=$(shell seq 0 3)
prefix:=final-
trainprefix:=train-
datfold:=/srv/abpid/dataset
NetDir:=/srv/abpid/Network_Models
Net:=$(NetDir)/ab.torch_net
Batch:=128

all: $(method)/sub.h5

sub : $(datfold)/sub.h5

model : $(Net)

data : $(seq:%=$(datfold)/$(trainprefix)%.h5)

$(datfold)/sub.h5 : $(datfold)/$(trainprefix)problem.h5
	python3 -u Inference.py $< -M $(Net) -o $@

$(Net) : $(seq:%=$(NetDir)/.Training_finished%)

$(NetDir)/.Training_finished% : $(datfold)/$(trainprefix)%.h5
	@mkdir -p $(dir $@)
	python3 -u Train_Nets.py $< -B $(Batch) -o $(Net) > $@.log 2>&1
	@touch $@

$(datfold)/$(trainprefix)problem.h5 : $(datfold)/$(prefix)problem.h5
	@mkdir -p $(dir $@)
	python3 -u Gen_Sets.py $^ -o $@

$(datfold)/$(trainprefix)%.h5 : $(datfold)/$(prefix)%.h5
	@mkdir -p $(dir $@)
	python3 -u Gen_Sets.py $^ -o $@

$(method)/sub.h5 : $(datfold)/$(prefix)problem.h5
	@mkdir -p $(dir $@)
	python3 -u thres.py $^ -o $@ > $@.log 2>&1

download : $(seq:%=$(datfold)/$(prefix)%.h5) $(datfold)/$(prefix)problem.h5
$(datfold)/$(prefix)%.h5 :
	@mkdir -p $(dir $@)
	wget http://hep.tsinghua.edu.cn/~orv/dc/$(notdir $@) -O $@
$(datfold)/$(prefix)problem.h5 :
	@mkdir -p $(dir $@)
	wget http://hep.tsinghua.edu.cn/~orv/dc/$(prefix)problem.h5 -O $@

.DELETE_ON_ERROR:
.SECONDARY:
