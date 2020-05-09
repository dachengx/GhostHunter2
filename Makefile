SHELL:=bash
method:=Threshold
seq:=$(shell seq 0 9)
prefix:=pre-
trainprefix:=train-
datfold:=/srv/abpid/dataset
NetDir:=/srv/abpid/Network_Models
PreTrained_Model = 

all: $(method)/sub.h5

model : $(NetDir)/ab.torch_net

$(NetDir)/ab.torch_net : $(NetDir)/.Training_finished
	@mkdir -p $(dir $@)
	python3 -u Choose_Nets.py $(dir $<) $@

$(NetDir)/.Training_finished : $(datfold)/$(trainprefix)0.h5 $(PreTrained_Model)
	@mkdir -p $(dir $@)
	python3 -u Train_Nets.py $< -B 32 -o $(dir $@) -P $(word 2,$^) > $@.log 2>&1
	@touch $@

$(datfold)/$(trainprefix)%.h5 : $(datfold)/$(prefix)%.h5
	@mkdir -p $(dir $@)
	python3 -u Gen_Sets.py $^ -o $@ > $@.log 2>&1

$(method)/sub.h5 : $(datfold)/$(prefix)problem.h5
	@mkdir -p $(dir $@)
	python3 -u thres.py $^ -o $@ > $@.log 2>&1

download : $(seq:%=$(datfold)/$(prefix)%.h5) $(datfold)/$(prefix)problem.h5
$(datfold)/$(prefix)%.h5 :
	@mkdir -p $(dir $@)
	wget http://hep.tsinghua.edu.cn/~orv/dc/$(nodir $@) -O $@
$(datfold)/$(prefix)problem.h5 :
	@mkdir -p $(dir $@)
	wget http://hep.tsinghua.edu.cn/~orv/dc/pre-problem.h5 -O $@

.DELETE_ON_ERROR:
.SECONDARY:
