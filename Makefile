SHELL := $(shell which bash)
LATEST_GIT_COMMIT := $(shell git log -1 --format=%h)
TIME := `/bin/date "+%Y-%m-%d-%H-%M-%S"`
USER?='pradip'

RUN_OS := LINUX
ifeq ($(OS),Windows_NT)
	RUN_OS = WIN32
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		RUN_OS = LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		RUN_OS = OSX
	endif
endif

CONDA_BIN = $(shell which conda)
CONDA_ROOT = $(shell $(CONDA_BIN) info --base)
CONDA_ENV_NAME ?= "tid"
CONDA_ENV_PREFIX = $(shell conda env list | grep $(CONDA_ENV_NAME) | sort | awk '{$$1=""; print $$0}' | tr -d '*\| ')
CONDA_ACTIVATE := source $(CONDA_ROOT)/etc/profile.d/conda.sh ; conda activate $(CONDA_ENV_NAME) && PATH=${CONDA_ENV_PREFIX}/bin:${PATH};	

conda_install:
	$(CONDA_BIN) env update -n $(CONDA_ENV_NAME) -f environment.yml
	$(CONDA_ACTIVATE) jupyter contrib nbextension install --user

install:
	pip install -U pip
	pip install --no-cache-dir -r requirements.txt
	pip install jupyter
	pip install jupyter_contrib_nbextensions
	jupyter contrib nbextension install --user

find_python:
	ps -ef | grep python

clean:
	-rm -rf ./**/.ipynb_checkpoints
	-rm -rf ./**/.pyc

STEM?=$(USER)_glasses
TAG?=latest
IMAGE:= $(STEM):$(TAG)
DOCKER_FILE=Dockerfile
GPU?='4,5'
dbuild:
	@echo "CREATING DOCKER IMAGE WITH NAME:" $(IMAGE)
	docker build -f $(DOCKER_FILE) \
				 -t $(IMAGE) .

CONTAINER_NAME?=$(USER)_glasses_exp
NB_PORT?= 8833
TB_PORT?= 8834
drun:
	@echo "CREATING DOCKER CONTAINER WITH NAME:" $(CONTAINER_NAME)
	@echo "USING DOCKER IMAGE WITH TAG:" $(IMAGE)
	@echo "WORKSPACE LOCATION:" $(WORKSPACE)
	docker run -i -t --rm \
					 --gpus all \
					 -v ~/.aws:/root/.aws:ro \
					 -v ${PWD}:/work \
					 -p $(NB_PORT):$(NB_PORT) \
					 -p $(TB_PORT):$(TB_PORT) \
					 --name ${CONTAINER_NAME} \
					 $(IMAGE) /bin/bash

docker_enter:
	docker exec -it ${CONTAINER_NAME} /bin/bash

docker_clean:
	docker rmi $(docker images --filter "dangling=true" -q --no-trunc) 2>/dev/null

jupyter:
	rm -rf nohup.out
	nohup jupyter notebook --port $(NB_PORT) --ip=* --no-browser --allow-root &
	sleep 5
	jupyter notebook list

jupyter-stop:
	jupyter notebook stop $(NB_PORT)

tb:
	$(eval EXP= taR_v7_insv0.1)
	@echo "FOR TENSORBOARD VIS EXP:" $(EXP)
	tensorboard --logdir "/workspace/model_weights/$(EXP)/model_dir/" --port $(TB_PORT)

train:
	$(eval EXP= retinaglass_mobile0.25_pad20)
	$(eval DATA= mtfl)
	$(eval VER= 1.0)
	@echo "FOR TRAINING, Data:" $(DATA)
	@echo "FOR TRAINING, Ver:" $(VER)
	@echo "FOR TRAINING, Exp:" $(EXP)
	python -u Pytorch_Retinaface/train.py \
				--training_dataset='./data/widerface/train/label.txt'  \
				--network='mobile0.25' \
				--num_workers=1 \
				--lr=1e-3 \
				--momentum=0.9 \
				--resume_net='./weights/mobilenet0.25_Final.pth' \
				--glass_pad=20 \
				--resume_epoch=0 \
				--weight_decay=5e-4 \
				--gamma=0.1 \
				--save_folder='./weights/$(EXP)/' > train-$(EXP)-$(DATA).out 2>&1

test-mtfl:
	$(test EXP= retinaglass_mobile0.25_pad20)
	$(test DATA= mtfl)
	@echo "Running Eval with Data:" $(DATA)-$(VER)
	@echo "Running Eval with Exp:" $(EXP)
	python Pytorch_Retinaface/test_mtfl.py \
				--trained_model='./weights/retinaglass_mobile0.25_pad20/mobilenet0.25_pad20_epoch_50.pth' \
				--network='mobile0.25' \
				--origin_size=True \
				--save_folder='./MTFL_evaluate/MTFL_txt/' \
				--test_datafolder='./data/MTFL' \
				--confidence_threshold=0.02 \
				--top_k=5000 \
				--nms_threshold=0.4 \
				--keep_top_k=750 \
				--vis_thres=0.2 \
				--save_image \
				--cpu > test-$(EXP)-$(DATA).out 2>&1

test-sample:
	$(test EXP= retinaglass_mobile0.25_pad20)
	$(test DATA= sample)
	@echo "Running Eval with Data:" $(DATA)-$(VER)
	@echo "Running Eval with Exp:" $(EXP)
	python Pytorch_Retinaface/test_sample.py \
				--trained_model='./weights/retinaglass_mobile0.25_pad20/mobilenet0.25_pad20_epoch_60.pth' \
				--network='mobile0.25' \
				--origin_size=True \
				--save_folder='./sample_evaluate/sample_txt/' \
				--test_datafolder='./data/sample_test' \
				--confidence_threshold=0.8 \
				--top_k=1 \
				--nms_threshold=0.5 \
				--keep_top_k=1 \
				--vis_thres=0.5 \
				--save_image \
				--cpu