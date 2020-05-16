SHELL:=bash
method:=Threshold
seq:=$(shell seq 0 1)
prefix:=pre-
trainprefix:=train-
datfold:=/srv/abpid/dataset
NetDir:=/srv/abpid/Network_Models
PreTrain:=net.txt

all: $(method)/sub.h5

sub : model $(datfold)/sub.h5

model : $(seq:%=$(NetDir)/ab.torch_net%)

data : $(seq:%=$(datfold)/$(trainprefix)%.h5)

$(datfold)/sub.h5 : $(datfold)/$(trainprefix)problem.h5 $(NetDir)/ab.torch_net$(word $(words $(seq)), $(seq))
	python3 -u Inference.py $< -M $(word 2,$^) -o $@

define train
$(NetDir)/ab.torch_net% : $(NetDir)/.Training_finished%
	python3 -u Choose_Nets.py $$< -P $$(PreTrain) -o $$@

$(NetDir)/.Training_finished% : $(datfold)/$(trainprefix)%.h5
	@mkdir -p $$(dir $$@)
	python3 -u Train_Nets.py $$< -B 32 -o $$(dir $$@) -P $$(PreTrain) > $$@.log 2>&1
	@touch $$@
endef
$(foreach i,$(seq),$(eval $(call train,$(i))))

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
	wget http://hep.tsinghua.edu.cn/~orv/dc/pre-problem.h5 -O $@

.DELETE_ON_ERROR:
.SECONDARY:
