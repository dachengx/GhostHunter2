SHELL:=bash
seq:=$(shell seq 0 0)
datfold:=/srv/abpid/dataset

all:

$(datfold)/pre-0.h5:
	@mkdir -p $(dir $@)
	wget http://hep.tsinghua.edu.cn/~orv/dc/pre-0.h5 -O $@
$(datfold)/pre-1.h5:
	@mkdir -p $(dir $@)
	wget http://hep.tsinghua.edu.cn/~orv/dc/pre-1.h5 -O $@
$(datfold)/pre-problem.h5:
	@mkdir -p $(dir $@)
	wget http://hep.tsinghua.edu.cn/~orv/dc/pre-problem.h5 -O $@

.DELETE_ON_ERROR:
.SECONDARY:
