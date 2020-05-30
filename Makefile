SHELL:=bash
method:=Threshold
dataseq:=$(shell seq 0 3)
fragnum:=1
seq:=$(shell seq -w 00 01) $(shell seq -w 10 11) $(shell seq -w 20 21) $(shell seq -w 30 31)
fragseq:=$(shell seq 0 ${fragnum})
prefix:=final-
trainprefix:=train-
datfold:=/srv/abpid/dataset
NetDir:=/srv/abpid/Network_Models
Net:=$(NetDir)/ab.torch_net

all: $(method)/sub.h5

sub : model $(datfold)/sub.h5

model : $(Net)

data : $(dataseq:%=$(datfold)/$(trainprefix)%.h5)

$(datfold)/sub.h5 : $(datfold)/$(trainprefix)problem.h5 $(Net)
	python3 -u Inference.py $< -M $(word 2,$^) -o $@ -N $(fragnum)

$(Net) : $(seq:%=$(NetDir)/.Training_finished%)

define train
$(NetDir)/.Training_finished%$(1) : $(datfold)/$(trainprefix)%.h5
	@mkdir -p $$(dir $$@)
	python3 -u Train_Nets.py $$< -B 32 -o $(Net) -N $(fragnum) $(1) > $$@.log 2>&1
	@touch $$@
endef

$(foreach i,$(fragseq),$(eval $(call train,$(i))))

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
