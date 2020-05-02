SHELL:=bash
seq:=$(shell seq 0 9)
datfold:=/srv/abpid/dataset
pre_train:=$(seq:%=$(datfold)/pre-%.h5)

all: $(method)/sub.h5

$(method)/sub.h5: $(datfold)/pre-problem.h5
	@mkdir -p $(dir $@)
	python3 thres.py $^ -o $@

download: $(pre_train) $(datfold)/pre-problem.h5
$(datfold)/pre-%.h5:
	@mkdir -p $(dir $@)
	wget http://hep.tsinghua.edu.cn/~orv/dc/$(nodir $@) -O $@
$(datfold)/pre-problem.h5:
	@mkdir -p $(dir $@)
	wget http://hep.tsinghua.edu.cn/~orv/dc/pre-problem.h5 -O $@

.DELETE_ON_ERROR:
.SECONDARY:
