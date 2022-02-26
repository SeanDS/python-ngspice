# Detect number of threads to build with.
ifndef CPU_COUNT
	NUM_THREADS = 1
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		NUM_THREADS = $(shell nproc || 1)
	endif
	ifeq ($(UNAME_S),Darwin)
		NUM_THREADS = $(shell sysctl -n hw.physicalcpu)
	endif
	ifeq ($(findstring MSYS_NT,$(UNAME_S)),MSYS_NT)
		NUM_THREADS = ${NUMBER_OF_PROCESSORS}
	endif
else
	NUM_THREADS = $(CPU_COUNT)
endif

BUILD_CMD = python setup.py build_ext -j $(NUM_THREADS) --inplace

default:
	$(BUILD_CMD)

dist:
	python -m build

clean:
	rm -rf build/ dist/
	rm -rf cython_debug
	rm -rf ngspice/ngspice.cpp
	rm -rf ngspice/ngspice.*.so
	rm -rf ngspice/ngspice.*.dll

realclean: clean
	git clean -fX
