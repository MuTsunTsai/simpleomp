
MAKEFLAGS += -j12

CXX := em++

SRCF := src
TEMP := build/obj
TARGET := build/dist

SRC := $(wildcard $(SRCF)/*.cpp)
OBJ := $(patsubst $(SRCF)/%.cpp,$(TEMP)/%.o,$(SRC))
DEP := $(patsubst $(SRCF)/%.cpp,$(TEMP)/%.d,$(SRC))
LDFLAGS :=\
	-L../build/dist\
	-lsimpleomp\
	--proxy-to-worker\
	-sENVIRONMENT=worker\
	-sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency\
	-sINITIAL_MEMORY=10MB

OUT := index

# https://emscripten.org/docs/tools_reference/emcc.html
# https://github.com/emscripten-core/emscripten/blob/main/src/settings.js

CPPFLAGS := -I$(SRCF) -std=c++20
CXXFLAGS := -Oz -flto -fopenmp -pthread -sDISABLE_EXCEPTION_CATCHING=1

HTML := $(TARGET)/$(OUT).html

ifeq ($(OS),Windows_NT)
# Error is still possible in parallel running, so we add extra protection
	MK = -@if not exist "$(@D)" mkdir "$(@D)" 2> NUL
	RM = rmdir /s /q
else
	MK = @mkdir -p "$(@D)"
endif

.PHONY: all
all: $(HTML)

$(HTML): $(OBJ) makefile
	$(MK)
	@echo Compiling [33m$(HTML)[0m
	@$(LINK.cc) $(USRFLAGS) -o $@ $(OBJ)
	@echo [33mWebAssembly compile complete![0m

$(TEMP)/%.o: $(SRCF)/%.cpp
	$(MK)
	@echo Compiling [32m$(patsubst $(SRCF)/%,%,$<)[0m
	@$(COMPILE.cc) -MMD -c $< -o $@

# Ignoring old dependencies that were removed
%.h: ;
%.hpp: ;
%.d: ;

-include $(DEP)

.PHONY: clean
clean:
	@$(RM) "$(TEMP)"

.PHONY: serve
serve:
	@pnpx statikk --coi build/dist